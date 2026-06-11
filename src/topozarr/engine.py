"""Streaming downsample driver: zarr-python for I/O, topozarr-core for compute."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np
import psutil
import zarr
from topozarr_core import block_reduce

Region = tuple[slice, ...]

DEFAULT_MAX_REGION_BYTES = 256 * 1024 * 1024

# rough peak-memory multiplier per in-flight region: source block, contiguous
# copy, reduced output, and zarr codec buffers
REGION_MEM_FACTOR = 5


def default_max_workers(region_bytes: int) -> int:
    """Thread count bounded by CPU count and available memory.

    Peak memory is roughly ``workers * 5 * region_bytes``, so workers are
    capped at half the available RAM divided by that per-region footprint.
    """
    cpu = os.cpu_count() or 4
    mem_budget = psutil.virtual_memory().available // 2
    by_mem = mem_budget // max(1, REGION_MEM_FACTOR * region_bytes)
    return max(1, min(cpu * 2, int(by_mem)))


class RegionTimer:
    """Thread-safe accumulator of per-region timings.

    Seconds are summed across worker threads, so totals can exceed wall time;
    divide by the worker count for an average per-thread split. ``block_s``
    covers ``get_block`` (source read plus any reduction); ``reduce_s`` is the
    kernel portion of that, so read time is ``block_s - reduce_s``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.regions = 0
        self.block_s = 0.0
        self.reduce_s = 0.0
        self.write_s = 0.0

    def add(
        self, *, block_s: float = 0.0, reduce_s: float = 0.0, write_s: float = 0.0
    ) -> None:
        with self._lock:
            self.regions += 1 if block_s or write_s else 0
            self.block_s += block_s
            self.reduce_s += reduce_s
            self.write_s += write_s

    def as_dict(self) -> dict[str, float | int]:
        return {
            "regions": self.regions,
            "read_s": round(self.block_s - self.reduce_s, 3),
            "reduce_s": round(self.reduce_s, 3),
            "write_s": round(self.write_s, 3),
        }


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
    on_block: Callable[[Region, np.ndarray], None] | None = None,
    timer: RegionTimer | None = None,
) -> list[Future[None]]:
    """Write every region of ``dst`` via ``get_block`` on a thread pool.

    With an external ``executor``, tasks are submitted and the pending futures
    returned for the caller to drain; otherwise a pool of ``max_workers`` is
    created, drained, and an empty list returned.

    ``on_block``, if provided, is called with ``(region, block)`` after the
    block is materialized but before it is written.  The caller is responsible
    for thread-safety of any shared state mutated inside ``on_block`` (disjoint
    shard-aligned regions guarantee no two threads touch the same output slice).
    Time spent in ``on_block`` is reported as ``reduce_s`` in the timer so that
    ``read_s = block_s - reduce_s`` remains the pure read time.
    """

    def one(region: Region) -> None:
        t0 = perf_counter()
        block = get_block(region)
        t1 = perf_counter()
        if on_block is not None:
            on_block(region, block)
        t2 = perf_counter()
        dst[region] = block
        if timer is not None:
            fused_s = (t2 - t1) if on_block is not None else 0.0
            timer.add(
                block_s=(t1 - t0) + fused_s,
                reduce_s=fused_s,
                write_s=perf_counter() - t2,
            )
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
    on_block: Callable[[Region, np.ndarray], None] | None = None,
    timer: RegionTimer | None = None,
) -> list[Future[None]]:
    """Write a region-indexable array into ``dst`` region by region.

    ``values`` may be a numpy array or any lazy array supporting
    tuple-of-slices indexing (e.g. ``xr.Variable`` backed by zarr/icechunk);
    each region is materialized individually, keeping peak memory at
    ``max_workers x region_size``. Pass ``source_chunks`` to widen regions to
    the source chunk grid so each source chunk is decoded once.

    For in-memory ``values`` (``np.ndarray``) the contiguity copy is skipped:
    slicing returns a (possibly strided) view, and both the reduce kernel and
    zarr setitem accept strided input, so copying would only waste memory.
    """
    shape = copy_region_shape(
        dst.shards or dst.chunks,
        dst.shape,
        dst.dtype.itemsize,
        source_chunks,
        max_region_bytes,
    )
    if isinstance(values, np.ndarray):

        def get_block(region: Region) -> np.ndarray:
            return values[region]
    else:

        def get_block(region: Region) -> np.ndarray:
            return np.ascontiguousarray(values[region])

    return _write_regions(
        dst,
        get_block,
        max_workers,
        region_shape=shape,
        executor=executor,
        on_region=on_region,
        on_block=on_block,
        timer=timer,
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
    timer: RegionTimer | None = None,
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
        t0 = perf_counter()
        out = block_reduce(block, stride, method, fill_value, skipna)
        if timer is not None:
            timer.add(reduce_s=perf_counter() - t0)
        return out

    return _write_regions(
        dst, get_block, max_workers, executor=executor, on_region=on_region, timer=timer
    )
