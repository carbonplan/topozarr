# test_pyramid.py
import pytest
from topozarr.coarsen import create_pyramid


def test_pyramid_datatree_structure(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    levels = 3
    pyramid = create_pyramid(ds, levels=levels, spec="ndpyramid")

    # 0 is coarsest, 2 is original res
    assert set(pyramid.dt.children) == {"0", "1", "2"}

    # shapes go: 16 -> 8 -> 4
    assert pyramid.dt["2"].ds.elevation.shape == (16, 16)
    assert pyramid.dt["1"].ds.elevation.shape == (8, 8)
    assert pyramid.dt["0"].ds.elevation.shape == (4, 4)


def test_crs_enforcement(create_dataset):
    ds_no_crs = create_dataset(add_crs=False)

    with pytest.raises(ValueError, match="dataset is missing a crs"):
        create_pyramid(ds_no_crs, levels=2)


def test_custom_dimensions(create_dataset):
    ds = create_dataset(x_dim="lon", y_dim="lat")
    pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")

    assert "lon" in pyramid.dt["1"].ds.dims
    assert pyramid.dt["0"].ds.elevation.shape == (8, 8)
