import pytest
from topozarr.coarsen import create_pyramid


def test_pyramid_spec_compliance(create_dataset):
    levels = 3
    ds = create_dataset(nx=100, ny=100)
    pyramid = create_pyramid(ds, levels=levels)

    assert "multiscales" in pyramid.dt.attrs

    finest = pyramid.dt["0"].ds
    coarsest = pyramid.dt["2"].ds

    assert finest.sizes["x"] == 100
    assert coarsest.sizes["x"] == 100 // (2 ** (levels - 1))

    layout = pyramid.dt.attrs["multiscales"]["layout"]
    assert layout[0]["asset"] == "0"
    assert layout[0]["transform"]["scale"] == [1.0, 1.0]


def test_resampling_method_propagation(create_dataset):
    method = "max"
    pyramid = create_pyramid(create_dataset(), levels=2, method=method)
    assert pyramid.dt.attrs["multiscales"]["resampling_method"] == method


def test_encoding_contains_shards(create_dataset):
    ds = create_dataset(nx=1000, ny=1000)
    pyramid = create_pyramid(ds, levels=1)

    enc = pyramid.encoding["/0"]["elevation"]
    assert "chunks" in enc
    assert "shards" in enc
    assert all(shard >= chunk for shard, chunk in zip(enc["shards"], enc["chunks"]))


def test_zarr_conventions_array(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=2)

    assert "zarr_conventions" in pyramid.dt.attrs
    conventions = pyramid.dt.attrs["zarr_conventions"]

    convention_names = {conv["name"] for conv in conventions}
    assert "multiscales" in convention_names
    assert "proj:" in convention_names
    assert "spatial:" in convention_names


def test_translation_offsets(create_dataset):
    pyramid = create_pyramid(create_dataset(), levels=3)
    layout = pyramid.dt.attrs["multiscales"]["layout"]

    assert layout[0]["transform"]["translation"] == [0.0, 0.0]
    assert layout[1]["transform"]["translation"] == [0.5, 0.5]


def test_per_level_resampling_method(create_dataset):
    method = "mean"
    pyramid = create_pyramid(create_dataset(), levels=3, method=method)
    layout = pyramid.dt.attrs["multiscales"]["layout"]

    assert "resampling_method" not in layout[0]
    assert layout[1]["resampling_method"] == method


def test_spatial_root_attrs(create_dataset):
    nx, ny = 16, 16
    pyramid = create_pyramid(create_dataset(nx=nx, ny=ny), levels=2)
    attrs = pyramid.dt.attrs

    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert len(attrs["spatial:transform"]) == 6
    assert len(attrs["spatial:bbox"]) == 4
    assert attrs["spatial:shape"] == [ny, nx]

    # transform origin should be half a pixel before the first coordinate
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

    # each coarser level should have ~2x larger pixel size
    t0 = layout[0]["spatial:transform"]
    t1 = layout[1]["spatial:transform"]
    assert t1[0] == pytest.approx(t0[0] * 2)  # x_res doubles
    assert t1[4] == pytest.approx(t0[4] * 2)  # y_res doubles
    # origin (upper-left corner) stays the same
    assert t1[2] == pytest.approx(t0[2])
    assert t1[5] == pytest.approx(t0[5])

    # shape halves at each level
    assert layout[1]["spatial:shape"][0] == layout[0]["spatial:shape"][0] // 2
    assert layout[2]["spatial:shape"][0] == layout[1]["spatial:shape"][0] // 2


@pytest.mark.conformance
def test_geozarr_toolkit_validation(create_dataset):
    validate_attrs = pytest.importorskip("geozarr_toolkit").validate_attrs
    pyramid = create_pyramid(create_dataset(nx=32, ny=32), levels=3)
    errors = validate_attrs(pyramid.dt.attrs)
    for convention, errs in errors.items():
        assert errs == [], f"{convention} validation errors: {errs}"


def test_transform_dims(create_dataset):
    """Test transforms are created for each dimension"""
    ds = create_dataset().expand_dims(time=5)
    pyramid = create_pyramid(ds, levels=2, x_dim="x", y_dim="y")

    l1 = pyramid.dt.attrs["multiscales"]["layout"][1]["transform"]

    dim_map = dict(zip(ds.dims, zip(l1["scale"], l1["translation"])))

    # check scale and translation
    assert dim_map["time"] == (1.0, 0.0)
    assert dim_map["x"] == (2.0, 0.5)
    assert dim_map["y"] == (2.0, 0.5)
    assert len(l1["scale"]) == len(ds.dims)
