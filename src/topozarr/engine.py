"""Streaming downsample driver: zarr-python for I/O, topozarr-core for compute."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any

import numpy as np
import zarr
from topozarr_core import block_reduce

Region = tuple[slice, ...]

DEFAULT_MAX_REGION_BYTES = 256 * 1024 * 1024


def shard_aligned_regions(
    arr: zarr.Array, region_shape: tuple[int, ...] | None = None
) -> Iterator[Region]:
    """Iterate output regions aligned to the array's shard (or chunk) grid.

    ``region_shape`` overrides the grid; it must be a per-axis multiple of
    the shard size so writes never straddle partial shards.
    """
    if region_shape is None:
        region_shape = arr.shards or arr.chunks
    counts = [math.ceil(n / r) for n, r in zip(arr.shape, region_shape)]
    for idx in product(*(range(c) for c in counts)):
        yield tuple(
            slice(i * r, min((i + 1) * r, n))
            for i, r, n in zip(idx, region_shape, arr.shape)
        )


def _write_regions(
    dst: zarr.Array,
    get_block: Callable[[Region], np.ndarray],
    max_workers: int | None,
    region_shape: tuple[int, ...] | None = None,
) -> None:
    def one(region: Region) -> None:
        dst[region] = get_block(region)

    with ThreadPoolExecutor(max_workers) as ex:
        for _ in ex.map(one, shard_aligned_regions(dst, region_shape)):
            pass  # drain to surface exceptions


def _copy_region_shape(
    dst: zarr.Array,
    source_chunks: tuple[int, ...] | None,
    max_region_bytes: int,
) -> tuple[int, ...]:
    """Region shape for a source-to-dst copy: the dst shard grid, widened per
    axis to the lcm with the source chunk grid so each source chunk is read
    once. Falls back to the plain shard grid if the lcm region exceeds the
    memory budget.
    """
    shard = dst.shards or dst.chunks
    if source_chunks is None:
        return shard
    region = tuple(
        min(math.lcm(s, c), n) for s, c, n in zip(shard, source_chunks, dst.shape)
    )
    if math.prod(region) * dst.dtype.itemsize > max_region_bytes:
        return shard
    return region


def copy_array(
    values: Any,
    dst: zarr.Array,
    *,
    source_chunks: tuple[int, ...] | None = None,
    max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
    max_workers: int | None = None,
) -> None:
    """Write a region-indexable array into ``dst`` region by region.

    ``values`` may be a numpy array or any lazy array supporting
    tuple-of-slices indexing (e.g. ``xr.Variable`` backed by zarr/icechunk);
    each region is materialized individually, keeping peak memory at
    ``max_workers x region_size``. Pass ``source_chunks`` to widen regions to
    the source chunk grid so each source chunk is decoded once.
    """
    shape = _copy_region_shape(dst, source_chunks, max_region_bytes)
    _write_regions(
        dst,
        lambda region: np.ascontiguousarray(values[region]),
        max_workers,
        region_shape=shape,
    )


def downsample_level(
    src: zarr.Array,
    dst: zarr.Array,
    *,
    stride: tuple[int, ...],
    method: str,
    fill_value: float | int | None = None,
    skipna: bool = True,
    max_workers: int | None = None,
) -> None:
    """Block-reduce ``src`` into ``dst`` by integer ``stride`` per axis.

    Streams shard-sized output regions through ``topozarr_core.block_reduce``
    with reads/writes via zarr-python. Matches
    ``xarray.coarsen(boundary="trim")`` shape and skipna semantics.
    Arrays are limited to 4 dimensions (kernel limit).
    """
    if len(stride) != src.ndim or src.ndim != dst.ndim:
        raise ValueError(
            f"stride {stride} does not match src ndim {src.ndim} / dst ndim {dst.ndim}"
        )

    def get_block(region: Region) -> np.ndarray:
        in_sel = tuple(
            slice(s.start * f, min(s.stop * f, n))
            for s, f, n in zip(region, stride, src.shape)
        )
        block = np.ascontiguousarray(src[in_sel])
        return block_reduce(block, stride, method, fill_value, skipna)

    _write_regions(dst, get_block, max_workers)
