from __future__ import annotations

import math
from typing import Literal

DEFAULT_CHUNK_BYTES = 512 * 1024
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


def snap_chunk_to_source(
    dim_size: int,
    ideal_chunk: int,
    src_chunk: int,
    chunks_per_shard: int | None,
) -> int | None:
    """Chunk size near ``ideal_chunk`` whose shard (chunk * chunks_per_shard)
    nests with ``src_chunk``: the shard divides the source chunk or is a
    multiple of it, so copy regions cover whole source chunks.

    Returns None when no candidate chunk lies within [ideal/2, ideal*2] and
    >= 128 (caller falls back to the plain heuristic).
    """
    # small dims take a single chunk anyway; nothing to snap. src_chunk <= 0
    # shouldn't occur (callers derive it from real array chunk sizes) but is
    # guarded defensively since a bogus source chunk would otherwise divide
    # by zero below.
    if src_chunk <= 0 or dim_size <= 128 or dim_size <= ideal_chunk:
        return None
    cps = chunks_per_shard or 1
    ideal_shard = ideal_chunk * cps

    # candidate *shard* sizes that nest with the source chunk grid
    candidates: set[int] = set()
    # every divisor of src_chunk (found in pairs up to sqrt)
    for d in range(1, int(math.isqrt(src_chunk)) + 1):
        if src_chunk % d == 0:
            candidates.update((d, src_chunk // d))
    # multiples of src_chunk, up to the first one past 2x the ideal shard
    # (anything larger cannot yield a chunk within the [ideal/2, ideal*2] band)
    max_mult = max(1, (2 * ideal_shard) // src_chunk + 1)
    candidates.update(src_chunk * m for m in range(1, max_mult + 1))

    # keep shards that split evenly into cps chunks of acceptable size
    valid = [
        s // cps
        for s in candidates
        if s % cps == 0
        and s // cps >= 128
        and ideal_chunk / 2 <= s // cps <= ideal_chunk * 2
    ]
    if not valid:
        return None
    # closest to ideal; ties prefer the smaller chunk
    return min(valid, key=lambda c: (abs(c - ideal_chunk), c))
