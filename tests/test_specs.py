import pytest
from topozarr.coarsen import create_pyramid


@pytest.mark.parametrize("method", ["mean", "max", "min", "sum"])
def test_resampling_method(create_dataset, method):
    pyramid = create_pyramid(create_dataset(), levels=3, method=method)
    layout = pyramid.dt.attrs["multiscales"]["layout"]

    assert pyramid.dt.attrs["multiscales"]["resampling_method"] == method
    assert "resampling_method" not in layout[0]
    assert layout[1]["resampling_method"] == method


def test_translation_offsets(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=3)
    layout = pyramid.dt.attrs["multiscales"]["layout"]

    assert layout[0]["transform"]["translation"] == [0.0, 0.0]
    assert layout[1]["transform"]["translation"] == [0.5, 0.5]


def test_spatial_root_attrs(create_dataset):
    nx, ny = 16, 16
    pyramid = create_pyramid(create_dataset(nx=nx, ny=ny), levels=2)
    attrs = pyramid.dt.attrs

    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert len(attrs["spatial:transform"]) == 6
    assert len(attrs["spatial:bbox"]) == 4
    assert attrs["spatial:shape"] == [ny, nx]

    x_res, _, c, _, y_res, f = attrs["spatial:transform"]
    assert x_res == pytest.approx(1.0)
    assert y_res == pytest.approx(1.0)
    assert c == pytest.approx(-0.5)
    assert f == pytest.approx(-0.5)


def test_spatial_per_level_attrs(create_dataset):
    levels = 3
    pyramid = create_pyramid(create_dataset(nx=32, ny=32), levels=levels)
    layout = pyramid.dt.attrs["multiscales"]["layout"]

    for entry in layout:
        assert "spatial:transform" in entry
        assert len(entry["spatial:transform"]) == 6
        assert "spatial:shape" in entry
        assert len(entry["spatial:shape"]) == 2

    t0 = layout[0]["spatial:transform"]
    t1 = layout[1]["spatial:transform"]
    assert t1[0] == pytest.approx(t0[0] * 2)  # x_res doubles
    assert t1[4] == pytest.approx(t0[4] * 2)  # y_res doubles
    assert t1[2] == pytest.approx(t0[2])  # origin stays the same
    assert t1[5] == pytest.approx(t0[5])

    assert layout[1]["spatial:shape"][0] == layout[0]["spatial:shape"][0] // 2
    assert layout[2]["spatial:shape"][0] == layout[1]["spatial:shape"][0] // 2


def test_transform_dims(create_dataset):
    """Transforms are created for every dimension, including non-spatial ones."""
    ds = create_dataset().expand_dims(time=5)
    pyramid = create_pyramid(ds, levels=2, x_dim="x", y_dim="y")

    l1 = pyramid.dt.attrs["multiscales"]["layout"][1]["transform"]
    dim_map = dict(zip(ds.dims, zip(l1["scale"], l1["translation"])))

    assert dim_map["time"] == (1.0, 0.0)
    assert dim_map["x"] == (2.0, 0.5)
    assert dim_map["y"] == (2.0, 0.5)
    assert len(l1["scale"]) == len(ds.dims)


@pytest.mark.conformance
def test_geozarr_toolkit_detect_conventions(create_dataset):
    """geozarr conventions check"""
    geozarr_toolkit = pytest.importorskip("geozarr_toolkit")
    pyramid = create_pyramid(create_dataset(nx=32, ny=32), levels=3)
    detected = geozarr_toolkit.detect_conventions(pyramid.dt.attrs)
    assert "multiscales" in detected, "multiscales convention not detected"
    assert "spatial" in detected, "spatial convention not detected"
    assert "proj" in detected, "proj convention not detected"


@pytest.mark.conformance
def test_geozarr_toolkit_per_level_validation(create_dataset):
    """per level validation"""
    validate_attrs = pytest.importorskip("geozarr_toolkit").validate_attrs
    pyramid = create_pyramid(create_dataset(nx=32, ny=32), levels=3)
    for level_name, level_node in pyramid.dt.children.items():
        errors = validate_attrs(level_node.attrs)
        for convention, errs in errors.items():
            assert errs == [], (
                f"Level {level_name} {convention} validation errors: {errs}"
            )


@pytest.mark.conformance
def test_geozarr_toolkit_group_validation(create_dataset):
    """validate full pyramid conventions"""
    validate_group = pytest.importorskip("geozarr_toolkit").validate_group
    pyramid = create_pyramid(create_dataset(nx=32, ny=32), levels=3)
    errors = validate_group(pyramid.dt)
    for convention, errs in errors.items():
        assert errs == [], f"{convention} validation errors: {errs}"
