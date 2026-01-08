import pytest
import numpy as np
import xarray as xr
import xproj  # noqa ignore


@pytest.fixture
def create_dataset():
    def _generate(nx=16, ny=16, x_dim="x", y_dim="y", epsg="EPSG:4326", add_crs=True):
        ds = xr.Dataset(
            {"elevation": ((y_dim, x_dim), np.random.rand(ny, nx).astype("f4"))},
            coords={
                x_dim: np.linspace(0, nx - 1, nx),
                y_dim: np.linspace(0, ny - 1, ny),
            },
        )
        if add_crs:
            return ds.proj.assign_crs(spatial_ref=epsg)
        return ds

    return _generate
