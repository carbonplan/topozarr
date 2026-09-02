# Design

How topozarr turns an Xarray Dataset into a multiscale Zarr store, and
which knobs control memory and performance.

## Plan / execute split

[`create_pyramid`][topozarr.coarsen.create_pyramid] is pure planning — no data
 written. It produces a [`Pyramid`][topozarr.pyramid.Pyramid]
holding:

- **Datatree**: per-level `xr.Dataset`s with real (mean-coarsened)
  coordinates.
- **encoding**: chunk and shard sizes per variable per level.
- **attrs**: root metadata following the zarr-conventions
  [multiscales](https://github.com/zarr-conventions/multiscales),
  [proj](https://github.com/zarr-conventions/geo-proj), and
  [spatial](https://github.com/zarr-conventions/spatial) specs.

The level structure comes from either `levels` (dense `[1, 2, 4, ...]` factors)
or `factors` (explicit cumulative downsample factors, e.g. `[1, 4, 16]` for a
sparse pyramid). Either way the plan is the same shape.

There are two ways to materialize the plan:

- **`Pyramid.write`** (default): level 0 is streamed from the source dataset,
  then each level `N` is block-reduced from the already-written level `N - 1`
  through the Rust kernel (`topozarr_core.block_reduce`), so the source is read
  exactly once regardless of the number of levels. Work runs on a local thread
  pool (not Dask). The rest of this document describes this path.
- **`Pyramid.as_datatree`**: returns a lazy `xr.DataTree` (levels coarsened via
  `xarray.coarsen`) for Dask-distributed writes. You call `to_zarr` yourself,
  passing `pyramid.encoding`. It reproduces the kernel's semantics — each
  coarsen runs on an `f8` promotion with `_FillValue` masked to NaN, then the
  result is refilled, clipped and cast back to the source dtype — so both paths
  write the same bytes. A method with no `xarray.coarsen` equivalent (`nearest`
  aside, which has its own decimation path) raises `NotImplementedError` here
  rather than silently dispatching.

## Chunk and shard heuristics

Spatial dimensions aim for square chunks of `target_chunk_bytes` (default
~500 KB, sized for web visualization): the ideal chunk dim is
`sqrt(target_chunk_bytes / itemsize)` with a floor of 128, then evened out so
chunks divide the dimension as uniformly as possible. Non-spatial dimensions
(time, band, ...) always get chunk size 1; only their *shard* extent varies.

Shards group `chunks_per_shard` chunks per spatial dimension (default 4, i.e.
4×4 = 16 chunks, ~8 MB). Shards are also the unit of work during generation:
larger shards mean fewer, bigger reads/writes and more memory per worker.

`chunks_per_shard` sets a shard *byte budget* as well. A spatial dimension can
only hold as many chunks as fit whole, so a small raster — or any sufficiently
coarse pyramid level — leaves part of that budget unspent. The remainder widens
non-spatial dimensions instead of being discarded, innermost first, bounded by
each dimension's extent. Chunk size along those dimensions stays 1, so a reader
still fetches a single element rather than the whole group.

This makes the priority emergent rather than hardcoded: spatial dimensions are
sized first and non-spatial ones only ever receive what is left, so a large
raster (where spatial saturates the budget) is unaffected. It also lines up with
CF dimension order (`T, Z, Y, X`) without inspecting dimension names — `time` is
outermost, so fixed-cardinality dimensions like `band` or `return_period` pack
before it.

Bytes alone are not a sufficient bound. A shard index costs 16 bytes per inner
chunk no matter how small those chunks are, so at coarse levels the byte budget
would admit thousands of single-element chunks and the index fetch would come to
dominate the chunk it locates. The inner chunk count per shard is therefore
capped at `MAX_INNER_CHUNKS` (128), holding the index near 2 KB.

When the source dataset is itself chunked (zarr/icechunk/dask), chunk sizes are
*snapped* so the destination shard grid nests with the source chunk grid,
provided a candidate exists within a factor of 2 of the ideal chunk size. This
lets each source chunk be decoded exactly once during the copy.

A shard that *divides* the source chunk is preferred over one that is a multiple
of it, because that is also what xarray's `safe_chunks` requires of a Dask write:
the write unit must divide the dask block. A dividing shard rarely exists at the
requested `chunks_per_shard` — the 128-element chunk floor rejects the small
divisors — so `chunks_per_shard` is an *upper bound*, flexed down per spatial
dimension to the largest power of 2 that admits one. Only when no divisor works
at any of them does it fall back to a multiple, which still reads each source
chunk once but needs `safe_chunks=False` to write from Dask.

Levels above 0 have nothing to sniff: their templates are unchunked
placeholders. Their source chunking is derived instead — coarsening a dask array
by `factor` divides its block sizes by `factor` — so the snapping applies at
every level. The derivation holds only where the factor divides the block
evenly; 750 halves to 375 and then splits into (187, 94, 94, …) rather than 187
across the board, so an indivisible block falls back to the plain heuristic.

Alignment therefore runs out at coarse levels, once the block has shrunk below
half the ideal chunk size and no dividing shard is left in band.

## Streaming memory model

The unit of work is a shard-aligned **region** of the destination array.
Workers on a thread pool each read one region's input, reduce it, and write it
out; nothing larger than `workers x region` is ever in memory.

- **Level 0**: regions are widened per axis to `lcm(shard, source_chunk)` so
  whole source chunks are read once, unless that exceeds `max_region_bytes`
  (default 256 MB), in which case the plain shard grid is used.
- **Levels 1+**: the region is one output shard; the input block read from the
  previous level is the region scaled by the 2×2 stride (~4× larger).

Peak memory is roughly `max_workers * 5 * region_bytes` (source block,
contiguous copy, reduced output, codec buffers). With `max_workers=None` the
pool size is derived from that: `min(2 * cpu_count, mem_budget / (5 *
region_bytes))`, where the budget is half the available RAM. Pass an explicit
`max_workers` to override.

Levels are written sequentially — each one reads the previous — but all
variables within a level stream through one shared pool.

## Kernel semantics

`topozarr_core.block_reduce` (Rust, rayon-parallel, GIL released):

- methods: `mean`, `max`, `min`, `sum`, `nearest`, exported as
  `topozarr_core.METHODS` — the single source the Python layer validates
  `create_pyramid(method=...)` against, so a topozarr paired with a core
  that lacks a method fails at plan time rather than mid-write
- dtypes: `u8`, `u16`, `i16`, `i32`, `i64`, `f32`, `f64`
- 1–4 dimensional arrays
- shape follows `xarray.coarsen(boundary="trim")`: trailing partial windows
  are dropped; an axis smaller than its stride still yields one window
- `skipna=True` skips NaN and `_FillValue` elements; an all-missing window
  produces 0 for `sum` (matching `nansum`) and the fill value (or NaN) for
  `mean`/`max`/`min`
- integer dtypes stay integer: `mean` truncates toward zero (unlike
  `xarray.coarsen`, which promotes to float)
- `nearest` decimates: each window emits its top-left cell, ignoring
  `skipna`/`fill_value`. Intended for categorical data (class codes, masks)
  where averaging invents values; corner-pick is exactly composable, so
  chained per-step decimation equals decimation from native resolution

## Tuning knobs

| Knob | Where | Effect |
|------|-------|--------|
| `levels` / `factors` | `create_pyramid` | number of levels, or explicit cumulative downsample factors (sparse pyramids) |
| `target_chunk_bytes` | `create_pyramid` | chunk size on disk |
| `chunks_per_shard` | `create_pyramid` | shard size = work unit; `None` disables sharding |
| `max_region_bytes` | `Pyramid.write` | cap on level-0 region widening |
| `max_workers` | `Pyramid.write` | thread pool size; `None` = RAM/CPU-derived |
| `keep_levels_in_memory` | `Pyramid.write` | keep written levels in RAM to skip re-reads; `None` = auto when they fit |
| `progress` | `Pyramid.write` | tqdm bar over written regions |
