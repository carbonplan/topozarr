import numpy as np
import pytest
import xarray as xr
from hypothesis import given, strategies as st
from topozarr.coarsen import create_pyramid

spatial_names = st.sampled_from(["x", "y", "lon", "lat"])
extra_names = st.sampled_from(["time", "band"])


@st.composite
def heterogeneous_datasets(draw):
    """Generates datasets with spatial and extra dims ( 1 to 10)."""
    x_n = draw(spatial_names)
    y_n = draw(spatial_names.filter(lambda x: x != x_n))
    nx, ny = draw(st.integers(1, 10)), draw(st.integers(1, 10))

    extras = draw(st.dictionaries(extra_names, st.integers(1, 3), max_size=1))

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
            enc = pyramid.encoding[path]["elevation"]
            for c, s in zip(enc["chunks"], enc["shards"]):
                assert c >= 1
                assert s % c == 0
