from topozarr.chunking import calculate_chunk_size, calculate_shard_size, get_ideal_dim


def test_chunk_and_shard_logic():
    itemsize = 4
    ideal_chunk = get_ideal_dim(itemsize, 1 * 1024 * 1024)
    ideal_shard = get_ideal_dim(itemsize, 10 * 1024 * 1024)

    dim_size = 2000

    chunk_size = calculate_chunk_size(dim_size, ideal_chunk)
    assert chunk_size == 500

    shard_size = calculate_shard_size(dim_size, chunk_size, ideal_shard)
    assert shard_size == 1500
    assert shard_size % chunk_size == 0


def test_small_dimension_handling():
    size = 100
    c = calculate_chunk_size(size, 512)
    assert c == 100
    s = calculate_shard_size(size, c, 1618)
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

    target = 1 * 1024 * 1024
    pyramid = create_pyramid(
        ds, levels=1, target_chunk_bytes=target, target_shard_bytes=target
    )

    enc = pyramid.encoding["/0"]["elevation"]
    assert enc["chunks"] == enc["shards"]
