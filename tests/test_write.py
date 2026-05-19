import numpy as np
import pytest
import zarr
import xarray as xr
from topozarr.coarsen import create_pyramid
from topozarr.write import write_pyramid


def test_write_pyramid_structure(create_dataset, tmp_path):
    """write_pyramid writes all levels and the root multiscale attrs."""
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)
    store = str(tmp_path / "out.zarr")

    write_pyramid(pyramid, store, zarr_format=3)

    root = zarr.open_group(store, mode="r")
    assert "multiscales" in dict(root.attrs)
    assert set(root.group_keys()) == {"0", "1", "2"}


def test_write_pyramid_shapes(create_dataset, tmp_path):
    """Each level's on-disk shape matches the pyramid's DataTree shape."""
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)
    store = str(tmp_path / "out.zarr")

    write_pyramid(pyramid, store, zarr_format=3)

    for i in range(3):
        level_ds = xr.open_zarr(store, group=str(i), consolidated=False)
        expected_shape = pyramid.dt[f"/{i}"].ds["elevation"].shape
        assert level_ds["elevation"].shape == expected_shape, (
            f"Level {i} shape mismatch: got {level_ds['elevation'].shape}, "
            f"expected {expected_shape}"
        )


def test_write_pyramid_values_consistent(create_dataset, tmp_path):
    """Values written by write_pyramid are close to the chained-coarsen values."""
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=3)
    store = str(tmp_path / "out.zarr")

    write_pyramid(pyramid, store, zarr_format=3)

    written = xr.open_zarr(store, group="2", consolidated=False)["elevation"].values
    expected = pyramid.dt["/2"].ds["elevation"].compute().values
    np.testing.assert_allclose(written, expected, rtol=1e-5)


def test_write_pyramid_root_attrs(create_dataset, tmp_path):
    """Root-level spatial and multiscale attributes are written into zarr.json."""
    ds = create_dataset(nx=16, ny=16)
    pyramid = create_pyramid(ds, levels=2)
    store = str(tmp_path / "out.zarr")

    write_pyramid(pyramid, store, zarr_format=3)

    root_attrs = dict(zarr.open_group(store, mode="r").attrs)
    assert "spatial:transform" in root_attrs
    assert "spatial:bbox" in root_attrs
    assert "proj:code" in root_attrs
    assert "multiscales" in root_attrs


def test_write_pyramid_level_attrs(create_dataset, tmp_path):
    """Dataset-level attributes are preserved through write_pyramid."""
    ds = create_dataset(nx=16, ny=16)
    ds.attrs["title"] = "test dataset"
    pyramid = create_pyramid(ds, levels=2)
    store = str(tmp_path / "out.zarr")

    write_pyramid(pyramid, store, zarr_format=3)

    for i in range(2):
        level_ds = xr.open_zarr(store, group=str(i), consolidated=False)
        assert level_ds.attrs.get("title") == "test dataset"


def test_write_pyramid_encoding(create_dataset, tmp_path):
    """Encoding (dtype override) is applied when writing."""
    ds = create_dataset(nx=32, ny=32)
    pyramid = create_pyramid(ds, levels=2)

    for level_enc in pyramid.encoding.values():
        for var_enc in level_enc.values():
            var_enc["dtype"] = "int16"
            var_enc["_FillValue"] = np.iinfo(np.int16).min

    store = str(tmp_path / "out.zarr")
    write_pyramid(pyramid, store, zarr_format=3)

    # xarray decodes int16+_FillValue back to float32 on read; check zarr directly
    z = zarr.open_group(store, mode="r")
    assert z["0"]["elevation"].dtype == np.int16


def test_write_pyramid_custom_dims(create_dataset, tmp_path):
    """write_pyramid respects custom x_dim/y_dim/method arguments."""
    ds = create_dataset(x_dim="lon", y_dim="lat")
    pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat", method="max")
    store = str(tmp_path / "out.zarr")

    write_pyramid(pyramid, store, x_dim="lon", y_dim="lat", method="max", zarr_format=3)

    root = zarr.open_group(store, mode="r")
    assert set(root.group_keys()) == {"0", "1"}
