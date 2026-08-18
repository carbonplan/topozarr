import numpy as np
import pytest
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
import zarr

from topozarr import (
    ZarrLayerVarConfig,
    attach_geozarr_metadata,
    create_pyramid,
    recommend_encoding,
)

SPEC_KEYS = {
    "zarr_conventions",
    "proj:code",
    "proj:wkt2",
    "spatial:dimensions",
    "spatial:registration",
    "spatial:transform",
    "spatial:bbox",
    "spatial:shape",
}


def test_attach_geozarr_metadata(create_dataset):
    ds = create_dataset(nx=8, ny=4)
    out = attach_geozarr_metadata(ds)

    assert SPEC_KEYS <= set(out.attrs)
    assert "multiscales" not in out.attrs
    assert [c["name"] for c in out.attrs["zarr_conventions"]] == ["proj", "spatial"]
    assert out.attrs["proj:code"] == "EPSG:4326"
    assert out.attrs["spatial:dimensions"] == ["y", "x"]
    assert out.attrs["spatial:shape"] == [4, 8]
    assert out.attrs["spatial:transform"] == [1.0, 0.0, -0.5, 0.0, 1.0, -0.5]
    assert out.attrs["spatial:bbox"] == [-0.5, -0.5, 7.5, 3.5]
    # input untouched
    assert "proj:code" not in ds.attrs


def test_attach_geozarr_metadata_explicit_crs(create_dataset):
    ds = create_dataset(add_crs=False)
    out = attach_geozarr_metadata(ds, crs="EPSG:3857")
    assert out.attrs["proj:code"] == "EPSG:3857"


def test_attach_geozarr_metadata_missing_crs(create_dataset):
    ds = create_dataset(add_crs=False)
    with pytest.raises(ValueError, match="missing a crs"):
        attach_geozarr_metadata(ds)


def test_attach_geozarr_metadata_custom_dims_and_hints(create_dataset):
    ds = create_dataset(x_dim="lon", y_dim="lat")
    hints = {"elevation": ZarrLayerVarConfig(colormap="blues", clim=[0.0, 1.0])}
    out = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat", layer_hints=hints)
    assert out.attrs["spatial:dimensions"] == ["lat", "lon"]
    assert out.attrs["zarr-layer"]["elevation"]["colormap"] == "blues"


def test_attach_geozarr_metadata_roundtrip_zarr(create_dataset, tmp_path):
    ds = attach_geozarr_metadata(create_dataset())
    path = tmp_path / "flat.zarr"
    ds.to_zarr(path, zarr_format=3, consolidated=False)

    root = zarr.open_group(path, mode="r")
    assert SPEC_KEYS <= set(root.attrs)
    reopened = xr.open_zarr(path, consolidated=False)
    xr.testing.assert_identical(
        reopened.elevation.drop_vars("spatial_ref"),
        ds.elevation.drop_vars("spatial_ref"),
    )


# --- recommend_encoding -----------------------------------------------------
# create_dataset(nx=2000, ny=2000) is a float32 raster big enough that the
# chunk heuristic actually splits it.


def test_recommend_encoding_shape(create_dataset):
    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    ds["mask"] = ((), np.float32(0))  # non-spatial: no encoding
    enc = recommend_encoding(ds)

    assert set(enc) == {"elevation"}
    assert set(enc["elevation"]) == {"chunks", "shards"}
    chunks, shards = enc["elevation"]["chunks"], enc["elevation"]["shards"]
    assert len(chunks) == len(shards) == 2
    assert all(s % c == 0 for s, c in zip(shards, chunks))


def test_recommend_encoding_no_sharding(create_dataset):
    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    assert set(recommend_encoding(ds, chunks_per_shard=None)["elevation"]) == {"chunks"}


@pytest.mark.parametrize("bad", [3, 64])
def test_recommend_encoding_invalid_chunks_per_shard(create_dataset, bad):
    with pytest.raises(ValueError, match="chunks_per_shard must be one of"):
        recommend_encoding(create_dataset(add_crs=False), chunks_per_shard=bad)


def test_recommend_encoding_missing_dims(create_dataset):
    ds = create_dataset()
    with pytest.raises(ValueError, match="x_dim 'lon' not found"):
        recommend_encoding(ds, x_dim="lon")
    with pytest.raises(ValueError, match="y_dim 'lat' not found"):
        recommend_encoding(ds, y_dim="lat")


def test_recommend_encoding_no_spatial_variable(create_dataset):
    ds = create_dataset().rename({"elevation": "keep"})
    ds["keep"] = ds.keep.isel(y=0)  # drops the y dim
    with pytest.raises(ValueError, match="nothing to encode"):
        recommend_encoding(ds)


def test_recommend_encoding_needs_no_crs(create_dataset):
    ds = create_dataset(add_crs=False)
    assert "elevation" in recommend_encoding(ds)


def test_recommend_encoding_smaller_budget_smaller_chunks(create_dataset):
    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    big = recommend_encoding(ds)["elevation"]["chunks"]
    small = recommend_encoding(ds, target_chunk_bytes=64 * 1024)["elevation"]["chunks"]
    assert all(s < b for s, b in zip(small, big))


def test_recommend_encoding_custom_dims_and_nonspatial(create_dataset):
    ds = create_dataset(
        nx=500, ny=500, x_dim="lon", y_dim="lat", extra_dims={"time": 4}, add_crs=False
    )
    enc = recommend_encoding(ds, x_dim="lon", y_dim="lat")["elevation"]

    time_idx = ds.elevation.get_axis_num("time")
    # chunks stay at 1 along time so readers still fetch one step at a time,
    # while the shard spends leftover budget widening it
    assert enc["chunks"][time_idx] == 1
    assert enc["shards"][time_idx] > 1


def test_recommend_encoding_snaps_to_source_chunks(create_dataset):
    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    store = zarr.storage.MemoryStore()
    ds.to_zarr(
        store, consolidated=False, encoding={"elevation": {"chunks": (1000, 1000)}}
    )
    lazy = xr.open_dataset(store, engine="zarr", chunks=None, consolidated=False)

    enc = recommend_encoding(lazy)["elevation"]
    for shard in enc["shards"]:
        assert 1000 % shard == 0 or shard % 1000 == 0
    # unchunked source keeps the pure heuristic, so the two differ
    assert enc != recommend_encoding(ds)["elevation"]


def test_recommend_encoding_ignores_disagreeing_source_chunks(create_dataset):
    """Spatial vars chunked differently -> no single source to snap to."""
    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    ds["other"] = ds.elevation.copy()
    mixed = ds.chunk({"y": 1000, "x": 1000})
    mixed["other"] = mixed.other.chunk({"y": 500, "x": 500})

    assert recommend_encoding(mixed) == recommend_encoding(ds)
    # agreeing chunks still snap
    assert recommend_encoding(ds.chunk({"y": 1000, "x": 1000})) != recommend_encoding(
        ds
    )


def test_recommend_encoding_dask_source_needs_shard_aligned_chunks(create_dataset):
    """safe_chunks compares dask blocks to the shard, so dask needs a rechunk.

    Pins the caveat the docstring documents: snapping aligns chunks, not
    shards, so writing a dask-backed dataset with the recommendation as-is
    raises unless the source is rechunked to the shards first.
    """
    ds = create_dataset(nx=2000, ny=2000, add_crs=False).chunk({"y": 750, "x": 750})
    enc = recommend_encoding(ds)
    shards = enc["elevation"]["shards"]

    with pytest.raises(ValueError, match="would overlap multiple Dask chunks"):
        ds.to_zarr(
            zarr.storage.MemoryStore(),
            zarr_format=3,
            consolidated=False,
            encoding=enc,
        )

    rechunked = ds.chunk(dict(zip(ds.elevation.dims, shards)))
    rechunked.to_zarr(
        zarr.storage.MemoryStore(), zarr_format=3, consolidated=False, encoding=enc
    )
    ds.to_zarr(
        zarr.storage.MemoryStore(),
        zarr_format=3,
        consolidated=False,
        encoding=enc,
        safe_chunks=False,
    )


def test_recommend_encoding_roundtrip_zarr(create_dataset, tmp_path):
    ds = create_dataset(nx=2000, ny=2000)
    enc = recommend_encoding(ds)
    path = tmp_path / "flat.zarr"
    attach_geozarr_metadata(ds).to_zarr(
        path, zarr_format=3, consolidated=False, encoding=enc
    )

    arr = zarr.open_group(path, mode="r")["elevation"]
    assert arr.shards == enc["elevation"]["shards"]
    assert arr.chunks == enc["elevation"]["chunks"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"levels": 1},
        {"levels": 3},
        {"factors": [1, 4, 16]},
        {"levels": 2, "chunks_per_shard": None},
        {"levels": 2, "target_chunk_bytes": 128 * 1024},
    ],
)
@pytest.mark.parametrize("chunked_source", [False, True])
def test_pyramid_level0_encoding_matches_recommend_encoding(
    create_dataset, kwargs, chunked_source
):
    """create_pyramid delegates level 0 to recommend_encoding, source sniff and all."""
    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    if chunked_source:
        store = zarr.storage.MemoryStore()
        ds.to_zarr(
            store, consolidated=False, encoding={"elevation": {"chunks": (1000, 1000)}}
        )
        ds = xr.open_dataset(store, engine="zarr", chunks=None, consolidated=False)
    ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

    enc_kwargs = {k: v for k, v in kwargs.items() if k not in ("levels", "factors")}
    pyramid = create_pyramid(ds, **kwargs)
    assert pyramid.encoding["/0"] == recommend_encoding(ds, **enc_kwargs)
