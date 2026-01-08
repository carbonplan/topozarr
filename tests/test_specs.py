from topozarr.coarsen import create_pyramid

import pytest


@pytest.mark.parametrize("spec_name", ["ndpyramid", "zarr-multiscales"])
def test_pyramid_spec_compliance(create_dataset, spec_name):
    levels = 3
    ds = create_dataset(nx=100, ny=100)
    pyramid = create_pyramid(ds, levels=levels, spec=spec_name)

    assert "multiscales" in pyramid.dt.attrs

    if spec_name == "ndpyramid":
        # 0 is coarsest, 2 is finest
        coarsest = pyramid.dt["0"].ds
        finest = pyramid.dt["2"].ds
    else:
        # 0 is finest , 2 is coarsest
        finest = pyramid.dt["0"].ds
        coarsest = pyramid.dt["2"].ds

    assert finest.sizes["x"] == 100
    assert coarsest.sizes["x"] == 100 // (2 ** (levels - 1))

    if spec_name == "zarr-multiscales":
        layout = pyramid.dt.attrs["multiscales"]["layout"]
        assert layout[0]["asset"] == "0"
        assert layout[0]["transform"]["scale"] == [1.0, 1.0]
    else:
        datasets = pyramid.dt.attrs["multiscales"][0]["datasets"]
        assert datasets[0]["path"] == "2"


def test_resampling_method_propagation(create_dataset):
    method = "max"
    pyramid = create_pyramid(
        create_dataset(), levels=2, method=method, spec="zarr-multiscales"
    )
    assert pyramid.dt.attrs["multiscales"]["resampling_method"] == method


def test_encoding_contains_shards(create_dataset):
    ds = create_dataset(nx=1000, ny=1000)
    pyramid = create_pyramid(ds, levels=1)

    enc = pyramid.encoding["0"]["elevation"]
    assert "chunks" in enc
    assert "shards" in enc
    assert all(shard >= chunk for shard, chunk in zip(enc["shards"], enc["chunks"]))
