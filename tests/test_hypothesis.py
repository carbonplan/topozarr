import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given, settings, strategies as st
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
    should_fail = min_dim < (2 ** (levels - 1))

    if should_fail:
        with pytest.raises(ValueError, match="cannot coarsen"):
            create_pyramid(ds, levels=levels, x_dim=x_dim, y_dim=y_dim)
    else:
        pyramid = create_pyramid(ds, levels=levels, x_dim=x_dim, y_dim=y_dim)
        assert len(pyramid.dt.children) == levels

        for path in pyramid.encoding:
            for var_name, enc in pyramid.encoding[path].items():
                for c, s in zip(enc["chunks"], enc["shards"]):
                    assert c >= 1
                    assert s % c == 0


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
    attrs = pyramid.dt.attrs
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
        level_ds = pyramid.dt[str(i)].ds
        assert entry["spatial:shape"] == [level_ds.sizes["y"], level_ds.sizes["x"]]

        # pixel size doubles per level, but only checkable when >1 pixel exists to
        # derive a resolution from coordinates (single-pixel levels default to 1.0)
        if level_ds.sizes["x"] > 1 and level_ds.sizes["y"] > 1:
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
