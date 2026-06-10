"""Streaming downsample driver: zarr-python for I/O, topozarr-core for compute."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import zarr
from topozarr_core import block_reduce

Region = tuple[slice, ...]


def shard_aligned_regions(arr: zarr.Array) -> Iterator[Region]:
    """Iterate output regions aligned to the array's shard (or chunk) grid."""
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
) -> None:
    def one(region: Region) -> None:
        dst[region] = get_block(region)

    with ThreadPoolExecutor(max_workers) as ex:
        for _ in ex.map(one, shard_aligned_regions(dst)):
            pass  # drain to surface exceptions


def copy_array(
    values: np.ndarray, dst: zarr.Array, *, max_workers: int | None = None
) -> None:
    """Write an in-memory array into ``dst`` region by region."""
    _write_regions(dst, lambda region: values[region], max_workers)


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
