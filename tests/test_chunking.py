from topozarr.chunking import (
    calculate_chunk_size,
    calculate_shard_size,
    get_ideal_dim,
    snap_chunk_to_source,
)


def test_chunk_and_shard_logic():
    itemsize = 4
    ideal_chunk = get_ideal_dim(itemsize, 1 * 1024 * 1024)

    dim_size = 2000

    chunk_size = calculate_chunk_size(dim_size, ideal_chunk)
    assert chunk_size == 500

    # chunks_per_shard=2 → 2 chunks per shard dim → shard=1000
    shard_size = calculate_shard_size(dim_size, chunk_size, 2)
    assert shard_size == 1000
    assert shard_size % chunk_size == 0


def test_small_dimension_handling():
    size = 100
    c = calculate_chunk_size(size, 512)
    assert c == 100
    # only 1 complete chunk fits, so shard is capped at 1 chunk regardless of chunks_per_shard
    s = calculate_shard_size(size, c, 4)
    assert s == 100


def test_chunk_size_overrides(create_dataset):
    from topozarr.coarsen import create_pyramid

    ds = create_dataset(nx=1000, ny=1000)

    pyramid_small = create_pyramid(ds, levels=1, target_chunk_bytes=1024)
    pyramid_large = create_pyramid(ds, levels=1, target_chunk_bytes=10 * 1024 * 1024)

    chunk_small = pyramid_small.encoding["/0"]["elevation"]["chunks"]
    chunk_large = pyramid_large.encoding["/0"]["elevation"]["chunks"]

    assert chunk_small[0] < chunk_large[0]


def test_shard_size_overrides(create_dataset):
    from topozarr.coarsen import create_pyramid

    ds = create_dataset(nx=1000, ny=1000)

    # chunks_per_shard=1 means shard dim == chunk dim
    pyramid = create_pyramid(ds, levels=1, chunks_per_shard=1)

    enc = pyramid.encoding["/0"]["elevation"]
    assert enc["chunks"] == enc["shards"]


def test_written_arrays_match_shard_encoding(create_dataset):
    import numpy as np
    import zarr

    from topozarr.coarsen import create_pyramid

    ds = create_dataset(nx=1000, ny=1000)
    pyramid = create_pyramid(ds, levels=2, target_chunk_bytes=1024)
    store = zarr.storage.MemoryStore()
    pyramid.write(store)

    root = zarr.open_group(store, mode="r")
    for level_path, level_encoding in pyramid.encoding.items():
        for var_name, var_enc in level_encoding.items():
            arr = root[f"{level_path.lstrip('/')}/{var_name}"]
            assert arr.chunks == var_enc["chunks"]
            assert arr.shards == var_enc["shards"]

    # metadata alone doesn't prove the write path stored real data: check
    # level 0 (a verbatim copy) actually holds the source values, not a
    # zero-filled or corrupted array
    np.testing.assert_array_equal(root["0/elevation"][:], ds.elevation.values)


def test_disable_sharding(create_dataset):
    from topozarr.coarsen import create_pyramid

    ds = create_dataset(nx=1000, ny=1000)
    pyramid = create_pyramid(ds, levels=1, chunks_per_shard=None)
    enc = pyramid.encoding["/0"]["elevation"]

    assert "chunks" in enc
    assert "shards" not in enc


def test_snap_chunk_divides_source():
    # src chunk 3600, ideal 362, cps 4 → shard 1200 divides 3600, chunk 300
    c = snap_chunk_to_source(20000, 362, 3600, 4)
    assert c == 300
    assert 3600 % (c * 4) == 0


def test_snap_chunk_multiple_of_source():
    # src chunk 100 smaller than shard → shard becomes a multiple of 100
    c = snap_chunk_to_source(20000, 362, 100, 4)
    assert c is not None
    assert (c * 4) % 100 == 0
    assert 181 <= c <= 724  # within [ideal/2, ideal*2]


def test_snap_chunk_no_candidate():
    # prime src chunk admits no nested shard within tolerance
    assert snap_chunk_to_source(20000, 362, 1009, 4) is None


def test_snap_chunk_no_sharding():
    # chunks_per_shard=None snaps the chunk itself
    c = snap_chunk_to_source(20000, 362, 3600, None)
    assert c == 360
    assert 3600 % c == 0


def test_snap_chunk_small_dim():
    # dim fits in one chunk → defer to heuristic
    assert snap_chunk_to_source(100, 362, 3600, 4) is None
    assert snap_chunk_to_source(300, 362, 3600, 4) is None


def test_pyramid_level0_chunks_snap_to_source(create_dataset):
    import xarray as xr
    import zarr

    from topozarr.coarsen import create_pyramid

    ds = create_dataset(nx=2000, ny=2000, add_crs=False)
    store = zarr.storage.MemoryStore()
    ds.to_zarr(
        store, consolidated=False, encoding={"elevation": {"chunks": (1000, 1000)}}
    )
    lazy = xr.open_dataset(store, engine="zarr", chunks=None, consolidated=False)
    lazy = lazy.proj.assign_crs(spatial_ref="EPSG:4326")

    pyramid = create_pyramid(lazy, levels=2)
    enc0 = pyramid.encoding["/0"]["elevation"]
    # level-0 shard nests with the 1000-px source chunks
    for shard in enc0["shards"]:
        assert 1000 % shard == 0 or shard % 1000 == 0
    # level 1 keeps the pure heuristic (no source chunk info)
    enc1 = pyramid.encoding["/1"]["elevation"]
    assert enc1["chunks"][0] == calculate_chunk_size(1000, get_ideal_dim(4, 512 * 1024))


def test_calculate_shard_size():
    """calculate_shard_size must return values divisible by chunk_size and <= dim_size."""
    test_cases = [
        (815, 408, 4),  # only 1 complete chunk fits → shard=408
        (128, 64, 4),  # 2 complete chunks fit, capped at 2 → shard=128
    ]

    for dim_size, chunk_size, chunks_per_shard in test_cases:
        shard = calculate_shard_size(dim_size, chunk_size, chunks_per_shard)

        assert shard % chunk_size == 0, (
            f"shard {shard} not divisible by chunk {chunk_size}"
        )

        assert shard <= dim_size, f"shard {shard} exceeds dim {dim_size}"
