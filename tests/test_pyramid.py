# test_pyramid.py
import numpy as np
import pytest
import xarray as xr
import zarr
from topozarr.coarsen import create_pyramid
from topozarr.metadata import ZarrLayerVarConfig


def test_pyramid_structure(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    levels = 3
    pyramid = create_pyramid(ds, levels=levels)

    # 0 is original res, 2 is coarsest
    assert set(pyramid.level_templates) == {0, 1, 2}
    assert pyramid.levels == levels

    # shapes go: 16 -> 8 -> 4
    assert pyramid.level_templates[0].elevation.shape == (16, 16)
    assert pyramid.level_templates[1].elevation.shape == (8, 8)
    assert pyramid.level_templates[2].elevation.shape == (4, 4)


def test_pyramid_write_roundtrip(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=3)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    assert set(dt.children) == {"0", "1", "2"}
    assert dt["0"].ds.elevation.shape == (16, 16)
    assert dt["1"].ds.elevation.shape == (8, 8)
    assert dt["2"].ds.elevation.shape == (4, 4)

    # level 0 is a verbatim copy; level 1 matches xarray coarsen
    np.testing.assert_array_equal(dt["0"].ds.elevation.values, ds.elevation.values)
    expected = ds.coarsen(x=2, y=2, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values, expected.elevation.values, rtol=1e-6
    )
    np.testing.assert_allclose(dt["1"].ds.x.values, expected.x.values)


def test_crs_enforcement(create_dataset):
    ds_no_crs = create_dataset(add_crs=False)

    with pytest.raises(ValueError, match="dataset is missing a crs"):
        create_pyramid(ds_no_crs, levels=2)


def test_custom_dimensions(create_dataset):
    ds = create_dataset(x_dim="lon", y_dim="lat")
    pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")

    assert "lon" in pyramid.level_templates[1].dims
    assert pyramid.level_templates[0].elevation.shape == (16, 16)


def test_write_invalid_levels(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)

    with pytest.raises(ValueError, match=r"invalid levels \[2, 5\]"):
        pyramid.write(zarr.storage.MemoryStore(), levels=[1, 2, 5])


def test_non_uniform_coords_raise(create_dataset):
    ds = create_dataset()
    ds = ds.assign_coords(x=ds.x.values**2)

    with pytest.raises(ValueError, match="'x' is not uniformly spaced"):
        create_pyramid(ds, levels=2)


def test_single_pixel_level_resolution(create_dataset):
    # 4x4 -> 2x2 -> 1x1: the coarsest level has length-1 coords, so its
    # resolution must come from level 0 (1.0 here) scaled by 2^level
    ds = create_dataset(nx=4, ny=4)
    pyramid = create_pyramid(ds, levels=3)

    layout = pyramid.attrs["multiscales"]["layout"]
    transform = layout[2]["spatial:transform"]
    assert transform[0] == 4.0  # x resolution
    assert transform[4] == 4.0  # y resolution


def test_single_value_coord_raises(create_dataset):
    ds = create_dataset(nx=1, ny=4)

    with pytest.raises(ValueError, match="cannot infer resolution"):
        create_pyramid(ds, levels=1)


def test_spatial_var_ndim_limit(create_dataset):
    ds = create_dataset()
    ds["stacked"] = ds.elevation.expand_dims(a=2, b=2, c=2)

    with pytest.raises(ValueError, match="supports at most 4"):
        create_pyramid(ds, levels=2)


def test_zarr_layer_metadata_written(create_dataset):
    ds = create_dataset()
    config = {"elevation": ZarrLayerVarConfig(clim=[0.0, 1.0], colormap="viridis")}
    pyramid = create_pyramid(ds, levels=2, layer_hints=config)

    zarr_layer = pyramid.attrs["zarr-layer"]
    assert zarr_layer["elevation"]["clim"] == [0.0, 1.0]
    assert zarr_layer["elevation"]["colormap"] == "viridis"
