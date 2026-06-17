# Design

How topozarr turns an Xarray Dataset into a multiscale Zarr store, and
which knobs control memory and performance.

## Plan / execute split

[`create_pyramid`][topozarr.coarsen.create_pyramid] is pure planning — no data
 written. It produces a [`Pyramid`][topozarr.pyramid.Pyramid]
holding:

- **Datatree**: per-level `xr.Dataset`s with real (mean-coarsened)
  coordinates.
- **encoding**: chunk and shard sizes per variable per level,
- **attrs**: root metadata following the zarr-conventions
  [multiscales](https://github.com/zarr-conventions/multiscales),
  [proj](https://github.com/zarr-conventions/geo-proj), and
  [spatial](https://github.com/zarr-conventions/spatial) specs.

The level structure comes from either `levels` (dense `[1, 2, 4, ...]` factors)
or `factors` (explicit cumulative downsample factors, e.g. `[1, 4, 16]` for a
sparse pyramid). Either way the plan is the same shape.

There are three ways to materialize the plan:

- **`Pyramid.write`** (default): level 0 is streamed from the source dataset,
  then each level `N` is block-reduced from the already-written level `N - 1`
  through the Rust kernel (`topozarr_core.block_reduce`), so the source is read
  exactly once regardless of the number of levels. Work runs on a local thread
  pool (not Dask). The rest of this document describes this path.
- **`Pyramid.write(..., io="rust")`**: same streaming model, but spatial-variable
  regions are encoded and stored natively in the Rust kernel instead of through
  zarr-python — often faster on object stores.
- **`Pyramid.as_datatree`**: returns a lazy `xr.DataTree` (levels coarsened via
  `xarray.coarsen`) for Dask-distributed writes. You call `to_zarr` yourself,
  passing `pyramid.encoding`.

## Chunk and shard heuristics

Spatial dimensions aim for square chunks of `target_chunk_bytes` (default
~500 KB, sized for web visualization): the ideal chunk dim is
`sqrt(target_chunk_bytes / itemsize)` with a floor of 128, then evened out so
chunks divide the dimension as uniformly as possible. Non-spatial dimensions
(time, band, ...) always get chunk size 1.

Shards group `chunks_per_shard` chunks per spatial dimension (default 4, i.e.
4×4 = 16 chunks, ~8 MB). Shards are also the unit of work during generation:
larger shards mean fewer, bigger reads/writes and more memory per worker.

When the source dataset is itself chunked (zarr/icechunk/dask), level-0 chunk
sizes are *snapped* so the destination shard grid nests with the source chunk
grid (the shard divides the source chunk or is a multiple of it), provided a
candidate exists within a factor of 2 of the ideal chunk size. This lets each
source chunk be decoded exactly once during the level-0 copy.

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

- methods: `mean`, `max`, `min`, `sum`
- dtypes: `u8`, `u16`, `i16`, `i32`, `i64`, `f32`, `f64`
- 1–4 dimensional arrays
- shape follows `xarray.coarsen(boundary="trim")`: trailing partial windows
  are dropped; an axis smaller than its stride still yields one window
- `skipna=True` skips NaN and `_FillValue` elements; an all-missing window
  produces 0 for `sum` (matching `nansum`) and the fill value (or NaN) for
  `mean`/`max`/`min`

## Tuning knobs

| Knob | Where | Effect |
|------|-------|--------|
| `levels` / `factors` | `create_pyramid` | number of levels, or explicit cumulative downsample factors (sparse pyramids) |
| `target_chunk_bytes` | `create_pyramid` | chunk size on disk |
| `chunks_per_shard` | `create_pyramid` | shard size = work unit; `None` disables sharding |
| `max_region_bytes` | `Pyramid.write` | cap on level-0 region widening |
| `max_workers` | `Pyramid.write` | thread pool size; `None` = RAM/CPU-derived |
| `keep_levels_in_memory` | `Pyramid.write` | keep written levels in RAM to skip re-reads; `None` = auto when they fit |
| `io` | `Pyramid.write` | `"python"` (default) or `"rust"` native write path |
| `progress` | `Pyramid.write` | tqdm bar over written regions |
