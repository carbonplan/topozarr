import math

DEFAULT_CHUNK_BYTES = 0.5 * 1024 * 1024
DEFAULT_SHARD_BYTES = 10 * 1024 * 1024


def get_ideal_dim(itemsize: int, target_bytes: int) -> int:
    return max(128, int(math.sqrt(target_bytes / itemsize)))


def calculate_chunk_size(dim_size: int, ideal_chunk_dim: int) -> int:
    if dim_size <= 128 or dim_size <= ideal_chunk_dim:
        return dim_size
    num_chunks = math.ceil(dim_size / ideal_chunk_dim)
    return max(128, math.ceil(dim_size / num_chunks))


def calculate_shard_size(dim_size: int, chunk_size: int, ideal_shard_dim: int) -> int:
    num_chunks_in_ideal = max(1, ideal_shard_dim // chunk_size)
    total_chunks = math.ceil(dim_size / chunk_size)
    chunks_per_shard = max(1, min(num_chunks_in_ideal, total_chunks))
    shard_size = chunks_per_shard * chunk_size

    if shard_size > dim_size:
        chunks_that_fit = dim_size // chunk_size
        if chunks_that_fit > 0:
            shard_size = chunks_that_fit * chunk_size
        else:
            shard_size = dim_size

    return shard_size
