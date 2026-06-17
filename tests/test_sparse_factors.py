# test_sparse_factors.py -- sparse / non-uniform pyramids via explicit factors
import numpy as np
import pytest
import xarray as xr
import zarr
from topozarr.coarsen import create_pyramid


def test_sparse_factors_structure_and_values(create_dataset):
    ds = create_dataset(nx=64, ny=64)
    pyramid = create_pyramid(ds, factors=[1, 4, 16])

    assert pyramid.factors == [1, 4, 16]
    assert pyramid.levels == 3
    assert pyramid.level_templates[0].elevation.shape == (64, 64)
    assert pyramid.level_templates[1].elevation.shape == (16, 16)
    assert pyramid.level_templates[2].elevation.shape == (4, 4)

    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)

    # level 0 verbatim; level 1 == coarsen(x=4, y=4)
    np.testing.assert_array_equal(dt["0"].ds.elevation.values, ds.elevation.values)
    expected1 = ds.coarsen(x=4, y=4, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["1"].ds.elevation.values, expected1.elevation.values, rtol=1e-6
    )
    expected2 = ds.coarsen(x=16, y=16, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["2"].ds.elevation.values, expected2.elevation.values, rtol=1e-6
    )


def test_sparse_factors_metadata(create_dataset):
    ds = create_dataset(nx=64, ny=64)
    attrs = create_pyramid(ds, factors=[1, 4, 16]).attrs
    layout = attrs["multiscales"]["layout"]

    # step 1->4: scale 4.0, translation (4-1)/2 = 1.5 on spatial axes
    scale1 = layout[1]["transform"]["scale"]
    trans1 = layout[1]["transform"]["translation"]
    assert scale1 == [4.0, 4.0]
    assert trans1 == [1.5, 1.5]

    native_res = abs(attrs["spatial:transform"][0])
    # per-level spatial:transform resolution == native * cumulative factor
    res1 = abs(layout[1]["spatial:transform"][0])
    res2 = abs(layout[2]["spatial:transform"][0])
    np.testing.assert_allclose(res1, native_res * 4, rtol=1e-6)
    np.testing.assert_allclose(res2, native_res * 16, rtol=1e-6)


def test_non_uniform_factors(create_dataset):
    ds = create_dataset(nx=64, ny=64)
    pyramid = create_pyramid(ds, factors=[1, 2, 8])  # steps 2 then 4

    assert pyramid.level_templates[1].elevation.shape == (32, 32)
    assert pyramid.level_templates[2].elevation.shape == (8, 8)

    layout = pyramid.attrs["multiscales"]["layout"]
    assert layout[1]["transform"]["scale"] == [2.0, 2.0]
    assert layout[1]["transform"]["translation"] == [0.5, 0.5]
    assert layout[2]["transform"]["scale"] == [4.0, 4.0]
    assert layout[2]["transform"]["translation"] == [1.5, 1.5]

    store = zarr.storage.MemoryStore()
    pyramid.write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)
    expected2 = ds.coarsen(x=8, y=8, boundary="trim").mean()
    np.testing.assert_allclose(
        dt["2"].ds.elevation.values, expected2.elevation.values, rtol=1e-6
    )


def test_region_bytes_scales_with_step(create_dataset):
    ds = create_dataset(nx=64, ny=64)
    pyramid = create_pyramid(ds, factors=[1, 4])
    max_rb = 1 << 30
    region = pyramid._region_shape(1, "elevation", max_rb)
    out_bytes = int(np.prod(region)) * ds.elevation.dtype.itemsize
    # step=4 -> input block is 4*4 = 16x the output region
    assert pyramid._region_bytes(1, "elevation", max_rb) == out_bytes * 16


def test_single_level_factors(create_dataset):
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, factors=[1])
    assert pyramid.levels == 1
    assert set(pyramid.level_templates) == {0}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"factors": [2, 4]}, "must start at 1"),
        ({"factors": [1, 3, 8]}, "integer multiple"),
        ({"factors": [1, 4, 2]}, "strictly increasing"),
        ({"factors": [1, -4]}, "positive ints"),
        ({"factors": [1, 2.0]}, "positive ints"),
        ({"factors": []}, "non-empty"),
        ({"levels": 2, "factors": [1, 2]}, "exactly one"),
        ({}, "exactly one"),
    ],
)
def test_factors_validation(create_dataset, kwargs, match):
    ds = create_dataset(nx=16, ny=16)
    with pytest.raises(ValueError, match=match):
        create_pyramid(ds, **kwargs)


def test_write_sparse(create_dataset):
    ds = create_dataset(nx=64, ny=64)

    store = zarr.storage.MemoryStore()
    create_pyramid(ds, factors=[1, 4, 16]).write(store)
    dt = xr.open_datatree(store, engine="zarr", consolidated=False)

    expected_sizes = {"0": 64, "1": 16, "2": 4}
    for lvl, size in expected_sizes.items():
        assert dt[lvl].ds.elevation.shape == (size, size)

    np.testing.assert_allclose(
        dt["0"].ds.elevation.values,
        ds.elevation.values,
        rtol=1e-5,
    )
