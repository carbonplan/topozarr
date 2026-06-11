"""Streaming downsample driver: zarr-python for I/O, topozarr-core for compute."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import product
from typing import Any

import numpy as np
import psutil
import zarr
from topozarr_core import block_reduce

Region = tuple[slice, ...]

DEFAULT_MAX_REGION_BYTES = 256 * 1024 * 1024

# rough peak-memory multiplier per in-flight region: source block, contiguous
# copy, reduced output, and zarr codec buffers
_REGION_MEM_FACTOR = 5


def default_max_workers(region_bytes: int) -> int:
    """Thread count bounded by CPU count and available memory.

    Peak memory is roughly ``workers * 5 * region_bytes``, so workers are
    capped at half the available RAM divided by that per-region footprint.
    """
    cpu = os.cpu_count() or 4
    mem_budget = psutil.virtual_memory().available // 2
    by_mem = mem_budget // max(1, _REGION_MEM_FACTOR * region_bytes)
    return max(1, min(cpu * 2, int(by_mem)))


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
    executor: ThreadPoolExecutor | None = None,
    on_region: Callable[[], None] | None = None,
) -> list[Future[None]]:
    """Write every region of ``dst`` via ``get_block`` on a thread pool.

    With an external ``executor``, tasks are submitted and the pending futures
    returned for the caller to drain; otherwise a pool of ``max_workers`` is
    created, drained, and an empty list returned.
    """

    def one(region: Region) -> None:
        dst[region] = get_block(region)
        if on_region is not None:
            on_region()

    regions = shard_aligned_regions(dst, region_shape)
    if executor is not None:
        return [executor.submit(one, region) for region in regions]
    with ThreadPoolExecutor(max_workers) as ex:
        for _ in ex.map(one, regions):
            pass  # drain to surface exceptions
    return []


def copy_region_shape(
    shard: tuple[int, ...],
    shape: tuple[int, ...],
    itemsize: int,
    source_chunks: tuple[int, ...] | None,
    max_region_bytes: int,
) -> tuple[int, ...]:
    """Region shape for a source-to-dst copy: the dst shard grid, widened per
    axis to the lcm with the source chunk grid so each source chunk is read
    once. Falls back to the plain shard grid if the lcm region exceeds the
    memory budget.
    """
    if source_chunks is None:
        return shard
    region = tuple(
        min(math.lcm(s, c), n) for s, c, n in zip(shard, source_chunks, shape)
    )
    if math.prod(region) * itemsize > max_region_bytes:
        return shard
    return region


def copy_array(
    values: Any,
    dst: zarr.Array,
    *,
    source_chunks: tuple[int, ...] | None = None,
    max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
    max_workers: int | None = None,
    executor: ThreadPoolExecutor | None = None,
    on_region: Callable[[], None] | None = None,
) -> list[Future[None]]:
    """Write a region-indexable array into ``dst`` region by region.

    ``values`` may be a numpy array or any lazy array supporting
    tuple-of-slices indexing (e.g. ``xr.Variable`` backed by zarr/icechunk);
    each region is materialized individually, keeping peak memory at
    ``max_workers x region_size``. Pass ``source_chunks`` to widen regions to
    the source chunk grid so each source chunk is decoded once.
    """
    shape = copy_region_shape(
        dst.shards or dst.chunks,
        dst.shape,
        dst.dtype.itemsize,
        source_chunks,
        max_region_bytes,
    )
    return _write_regions(
        dst,
        lambda region: np.ascontiguousarray(values[region]),
        max_workers,
        region_shape=shape,
        executor=executor,
        on_region=on_region,
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
    executor: ThreadPoolExecutor | None = None,
    on_region: Callable[[], None] | None = None,
) -> list[Future[None]]:
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

    return _write_regions(
        dst, get_block, max_workers, executor=executor, on_region=on_region
    )
