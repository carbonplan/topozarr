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
