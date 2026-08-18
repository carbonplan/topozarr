import numpy as np
import pytest
import xarray as xr
import zarr
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from topozarr import attach_geozarr_metadata, recommend_encoding
from topozarr.coarsen import create_pyramid

spatial_names = st.sampled_from(["x", "y", "lon", "lat", "X", "Y"])
extra_names = st.sampled_from(["time", "band", "z"])


@st.composite
def heterogeneous_datasets(draw):
    x_n = draw(spatial_names)
    y_n = draw(spatial_names.filter(lambda x: x != x_n))
    nx, ny = draw(st.integers(1, 10)), draw(st.integers(1, 10))

    extras = draw(st.dictionaries(extra_names, st.integers(1, 3), max_size=2))

    all_dims = list(extras.keys()) + [y_n, x_n]
    shape = tuple(list(extras.values()) + [ny, nx])

    ds = xr.Dataset(
        {"elevation": (all_dims, np.zeros(shape, dtype="f4"))},
        coords={
            x_n: np.arange(nx),
            y_n: np.arange(ny),
            **{k: np.arange(v) for k, v in extras.items()},
        },
    )
    return ds.proj.assign_crs(spatial_ref="EPSG:4326"), x_n, y_n


@st.composite
def multi_variable_datasets(draw):
    x_n = draw(spatial_names)
    y_n = draw(spatial_names.filter(lambda x: x != x_n))
    nx, ny = draw(st.integers(4, 16)), draw(st.integers(4, 16))

    num_vars = draw(st.integers(2, 4))
    data_vars = {}
    for i in range(num_vars):
        data_vars[f"var_{i}"] = ([y_n, x_n], np.zeros((ny, nx), dtype="f4"))

    ds = xr.Dataset(
        data_vars,
        coords={x_n: np.arange(nx), y_n: np.arange(ny)},
    )
    return ds.proj.assign_crs(spatial_ref="EPSG:4326"), x_n, y_n


@settings(deadline=2000)
@given(ds_info=heterogeneous_datasets(), levels=st.integers(1, 5))
def test_pyramid_integration_robustness(ds_info, levels):
    ds, x_dim, y_dim = ds_info

    min_dim = min(ds.sizes[x_dim], ds.sizes[y_dim])

    if min_dim < (2 ** (levels - 1)):
        with pytest.raises(ValueError, match="cannot coarsen"):
            create_pyramid(ds, levels=levels, x_dim=x_dim, y_dim=y_dim)
    elif min_dim == 1:
        # single-pixel source dim: no spacing to derive a transform from
        with pytest.raises(ValueError, match="cannot infer resolution"):
            create_pyramid(ds, levels=levels, x_dim=x_dim, y_dim=y_dim)
    else:
        pyramid = create_pyramid(ds, levels=levels, x_dim=x_dim, y_dim=y_dim)
        assert pyramid.levels == levels

        for path in pyramid.encoding:
            for var_name, enc in pyramid.encoding[path].items():
                for c, s in zip(enc["chunks"], enc["shards"]):
                    assert c >= 1
                    assert s % c == 0

        store = zarr.storage.MemoryStore()
        pyramid.write(store)
        root = zarr.open_group(store, mode="r")
        assert set(root.keys()) == {str(lvl) for lvl in range(levels)}


@st.composite
def spatial_grid_datasets(draw):
    nx = draw(st.integers(2, 32))
    ny = draw(st.integers(2, 32))
    x_res = draw(st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
    y_res = draw(st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False))
    x0 = draw(st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False))
    y0 = draw(st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False))

    ds = xr.Dataset(
        {"elevation": (("y", "x"), np.zeros((ny, nx), dtype="f4"))},
        coords={
            "x": x0 + np.arange(nx) * x_res,
            "y": y0 + np.arange(ny) * y_res,
        },
    )
    return ds.proj.assign_crs(spatial_ref="EPSG:4326"), nx, ny, x_res, y_res, x0, y0


@settings(deadline=2000)
@given(ds_info=spatial_grid_datasets(), levels=st.integers(1, 4))
def test_spatial_transform_invariants(ds_info, levels):
    """Affine transform, bbox, and per-level shape invariants hold for arbitrary grids."""
    ds, nx, ny, x_res, y_res, x0, y0 = ds_info
    assume(min(nx, ny) >= 2 ** (levels - 1))

    pyramid = create_pyramid(ds, levels=levels)
    attrs = pyramid.attrs
    layout = attrs["multiscales"]["layout"]

    # root spatial:shape matches dataset
    assert attrs["spatial:shape"] == [ny, nx]

    # bbox extent matches grid footprint
    xmin, ymin, xmax, ymax = attrs["spatial:bbox"]
    assert xmax - xmin == pytest.approx(x_res * nx, rel=1e-5)
    assert ymax - ymin == pytest.approx(y_res * ny, rel=1e-5)

    # transform origin is half a pixel before the first coordinate
    a, _, c, _, e, f = attrs["spatial:transform"]
    assert c == pytest.approx(x0 - 0.5 * x_res, rel=1e-5)
    assert f == pytest.approx(y0 - 0.5 * y_res, rel=1e-5)

    # per-level: spatial:shape matches actual level dataset shape, pixel size doubles
    for i, entry in enumerate(layout):
        level_ds = pyramid.level_templates[i]
        assert entry["spatial:shape"] == [level_ds.sizes["y"], level_ds.sizes["x"]]

        # pixel size doubles per level (single-pixel levels fall back to
        # level-0 resolution * 2^level)
        level_x_res = entry["spatial:transform"][0]
        assert level_x_res == pytest.approx(x_res * (2**i), rel=1e-5)


@settings(deadline=2000)
@given(ds_info=multi_variable_datasets(), levels=st.integers(1, 3))
def test_multi_variable_encoding(ds_info, levels):
    ds, x_dim, y_dim = ds_info

    pyramid = create_pyramid(ds, levels=levels, x_dim=x_dim, y_dim=y_dim)

    for level_idx in range(levels):
        level_encoding = pyramid.encoding[f"/{level_idx}"]
        assert len(level_encoding) == len(ds.data_vars)


@st.composite
def flat_datasets(draw):
    """Arbitrary single-resolution raster: any dim names, shapes, dtype, extras.

    No CRS and no coarsening, so this reaches shapes `create_pyramid` rejects
    (single-pixel dims have no resolution to infer a transform from).
    """
    x_n = draw(spatial_names)
    y_n = draw(spatial_names.filter(lambda n: n != x_n))
    nx, ny = draw(st.integers(1, 64)), draw(st.integers(1, 64))
    extras = draw(st.dictionaries(extra_names, st.integers(1, 3), max_size=2))
    dtype = draw(st.sampled_from(["u1", "i2", "f4", "f8"]))

    dims = {**extras, y_n: ny, x_n: nx}
    ds = xr.Dataset(
        {"elevation": (tuple(dims), np.zeros(tuple(dims.values()), dtype=dtype))},
        coords={
            x_n: np.arange(nx),
            y_n: np.arange(ny),
            **{k: np.arange(v) for k, v in extras.items()},
        },
    )
    return ds, x_n, y_n


@settings(deadline=2000, max_examples=300)
@given(
    ds_info=flat_datasets(),
    chunks_per_shard=st.sampled_from([None, 1, 2, 4, 8]),
)
def test_recommend_encoding_invariants(ds_info, chunks_per_shard):
    """recommend_encoding output is well-formed and accepted by to_zarr verbatim."""
    ds, x_dim, y_dim = ds_info
    enc = recommend_encoding(
        ds, x_dim=x_dim, y_dim=y_dim, chunks_per_shard=chunks_per_shard
    )

    assert set(enc) == {"elevation"}
    da = ds.elevation
    chunks = enc["elevation"]["chunks"]
    assert len(chunks) == da.ndim
    assert all(1 <= c <= s for c, s in zip(chunks, da.shape))
    # chunks stay at 1 along non-spatial dims; only the shard widens them
    for i, dim in enumerate(da.dims):
        if dim not in (x_dim, y_dim):
            assert chunks[i] == 1

    if chunks_per_shard is None:
        assert "shards" not in enc["elevation"]
    else:
        shards = enc["elevation"]["shards"]
        assert len(shards) == da.ndim
        assert all(s % c == 0 for s, c in zip(shards, chunks))
        assert all(s <= dim for s, dim in zip(shards, da.shape))

    store = zarr.storage.MemoryStore()
    ds.to_zarr(store, zarr_format=3, consolidated=False, encoding=enc)
    arr = zarr.open_group(store, mode="r")["elevation"]
    assert arr.chunks == chunks
    assert arr.shards == enc["elevation"].get("shards")


@settings(deadline=2000)
@given(ds_info=flat_datasets())
def test_flat_geozarr_attrs_invariants(ds_info):
    """attach_geozarr_metadata describes the grid and adds no multiscales."""
    ds, x_dim, y_dim = ds_info
    ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

    if min(ds.sizes[x_dim], ds.sizes[y_dim]) == 1:
        # single-pixel dim: no spacing to derive a transform from, same as
        # create_pyramid. recommend_encoding still works on these.
        with pytest.raises(ValueError, match="cannot infer resolution"):
            attach_geozarr_metadata(ds, x_dim=x_dim, y_dim=y_dim)
        return

    attrs = attach_geozarr_metadata(ds, x_dim=x_dim, y_dim=y_dim).attrs

    assert "multiscales" not in attrs
    assert attrs["spatial:shape"] == [ds.sizes[y_dim], ds.sizes[x_dim]]
    assert attrs["spatial:dimensions"] == [y_dim, x_dim]
    xmin, ymin, xmax, ymax = attrs["spatial:bbox"]
    assert xmin < xmax and ymin < ymax
