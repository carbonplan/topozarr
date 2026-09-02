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
from topozarr.metadata import (
    MULTISCALES_CONVENTION,
    PROJ_CONVENTION,
    SPATIAL_CONVENTION,
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


@pytest.mark.parametrize("src_chunk", [500, 750, 256, 1000])
def test_recommend_encoding_dask_source_writes_unaided(create_dataset, src_chunk):
    """The headline snippet works on a dask source, no safe_chunks=False.

    safe_chunks compares dask blocks to the zarr write unit -- the shard, when
    sharding is on -- and requires the shard to divide the block. chunks_per_shard
    is flexed down until one does.
    """
    ds = create_dataset(nx=2000, ny=2000, add_crs=False).chunk(
        {"y": src_chunk, "x": src_chunk}
    )
    enc = recommend_encoding(ds)
    shards = enc["elevation"]["shards"]
    assert all(src_chunk % s == 0 for s in shards)

    ds.to_zarr(
        zarr.storage.MemoryStore(), zarr_format=3, consolidated=False, encoding=enc
    )


def test_recommend_encoding_dask_source_unalignable_still_needs_escape_hatch(
    create_dataset,
):
    """A source chunk too small to divide into a >= 128 chunk keeps the old rule.

    128 admits no dividing shard in the [ideal/2, ideal*2] band at any
    chunks_per_shard, so the recommendation falls back to a shard that is a
    *multiple* of the source chunk: reads stay aligned, but a dask write of it
    needs the escape hatch. Both documented workarounds still apply.
    """
    ds = create_dataset(nx=2000, ny=2000, add_crs=False).chunk({"y": 128, "x": 128})
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


def test_flat_and_pyramid_root_attrs_agree(create_dataset):
    """Flat and pyramid roots emit the same geozarr block, modulo multiscales."""
    ds = create_dataset(nx=16, ny=16)
    flat = attach_geozarr_metadata(ds).attrs
    root = create_pyramid(ds, levels=2).attrs

    assert set(root) - set(flat) == {"multiscales"}
    assert set(flat) - set(root) == set()
    for key in set(flat) - {"zarr_conventions"}:
        assert flat[key] == root[key], key

    assert flat["zarr_conventions"] == [PROJ_CONVENTION, SPATIAL_CONVENTION]
    assert root["zarr_conventions"] == [
        MULTISCALES_CONVENTION,
        PROJ_CONVENTION,
        SPATIAL_CONVENTION,
    ]

    # root and level 0 share one transform computation
    layout = root["multiscales"]["layout"]
    assert layout[0]["spatial:transform"] == root["spatial:transform"]
    assert layout[0]["spatial:shape"] == root["spatial:shape"]


def test_pyramid_write_matches_unchunked_source(create_dataset):
    """Flexing moves pyramid.encoding for a chunked source; data must not move."""
    import numpy as np

    from topozarr import create_pyramid

    ds = create_dataset(nx=1000, ny=1000)
    reference = zarr.storage.MemoryStore()
    create_pyramid(ds, levels=3).write(reference)
    ref = zarr.open_group(reference, mode="r")

    chunked = ds.chunk({"y": 500, "x": 500})
    pyramid = create_pyramid(chunked, levels=3)
    assert all(500 % s == 0 for s in pyramid.encoding["/0"]["elevation"]["shards"])

    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    got = zarr.open_group(store, mode="r")
    for lvl in range(3):
        np.testing.assert_array_equal(
            got[f"{lvl}/elevation"][:], ref[f"{lvl}/elevation"][:]
        )


@pytest.mark.parametrize("src_chunk,levels", [(500, 2), (750, 2), (1000, 3)])
def test_pyramid_datatree_writes_dask_unaided(create_dataset, src_chunk, levels):
    """The Dask-distributed path writes with pyramid.encoding, no escape hatch.

    Levels above 0 have no source chunking to sniff -- their templates are
    unchunked placeholders -- so it is derived from the level-0 blocks and the
    cumulative factor.
    """
    from topozarr import create_pyramid

    ds = create_dataset(nx=2000, ny=2000).chunk({"y": src_chunk, "x": src_chunk})
    pyramid = create_pyramid(ds, levels=levels)

    for lvl in range(levels):
        block = src_chunk // 2**lvl
        for shard in pyramid.encoding[f"/{lvl}"]["elevation"]["shards"]:
            assert block % shard == 0, f"level {lvl} shard {shard} vs block {block}"

    pyramid.as_datatree().to_zarr(
        zarr.storage.MemoryStore(),
        zarr_format=3,
        consolidated=False,
        encoding=pyramid.encoding,
    )


def test_pyramid_dask_safety_ends_when_blocks_outrun_the_chunk_band(create_dataset):
    """Coarse levels can fall out of alignment, and that is not fixable here.

    A shard must divide the dask block, and a chunk must stay within a factor
    of 2 of the ideal (362 for f4 at the default target, so >= 181). Once
    coarsening has shrunk the block below that, no dividing shard qualifies:
    src 500 reaches block 125 at level 2. Such a write needs safe_chunks=False.
    """
    from topozarr import create_pyramid

    ds = create_dataset(nx=2000, ny=2000).chunk({"y": 500, "x": 500})
    pyramid = create_pyramid(ds, levels=3)

    assert all(250 % s == 0 for s in pyramid.encoding["/1"]["elevation"]["shards"])
    assert not all(125 % s == 0 for s in pyramid.encoding["/2"]["elevation"]["shards"])

    with pytest.raises(ValueError, match="would overlap multiple Dask chunks"):
        pyramid.as_datatree().to_zarr(
            zarr.storage.MemoryStore(),
            zarr_format=3,
            consolidated=False,
            encoding=pyramid.encoding,
        )
    pyramid.as_datatree().to_zarr(
        zarr.storage.MemoryStore(),
        zarr_format=3,
        consolidated=False,
        encoding=pyramid.encoding,
        safe_chunks=False,
    )
