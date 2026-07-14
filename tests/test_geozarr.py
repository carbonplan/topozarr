import pytest
import xarray as xr
import zarr

from topozarr import ZarrLayerVarConfig, attach_geozarr_metadata

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
