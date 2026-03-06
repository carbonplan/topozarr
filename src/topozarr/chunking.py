import math
from typing import Literal

DEFAULT_CHUNK_BYTES = 0.5 * 1024 * 1024
DEFAULT_CHUNKS_PER_SHARD = 4

ChunksPerShard = Literal[1, 2, 4, 8, 16, 32]
VALID_CHUNKS_PER_SHARD = {1, 2, 4, 8, 16, 32}


def validate_chunks_per_shard(chunks_per_shard: int) -> None:
    if chunks_per_shard not in VALID_CHUNKS_PER_SHARD:
        raise ValueError(
            f"chunks_per_shard must be one of {sorted(VALID_CHUNKS_PER_SHARD)}, got {chunks_per_shard}"
        )


def get_ideal_dim(itemsize: int, target_bytes: int) -> int:
    return max(128, int(math.sqrt(target_bytes / itemsize)))


def calculate_chunk_size(dim_size: int, ideal_chunk_dim: int) -> int:
    if dim_size <= 128 or dim_size <= ideal_chunk_dim:
        return dim_size
    num_chunks = math.ceil(dim_size / ideal_chunk_dim)
    return max(128, math.ceil(dim_size / num_chunks))


def calculate_shard_size(dim_size: int, chunk_size: int, chunks_per_shard: int) -> int:
    complete_chunks = max(1, dim_size // chunk_size)
    actual_chunks_per_shard = min(chunks_per_shard, complete_chunks)
    return actual_chunks_per_shard * chunk_size
