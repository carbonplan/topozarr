# Design

Details on how `topozarr` turns an Xarray Dataset into a multiscale Zarr store, and
which knobs control memory and performance.

## Plan / execute split

[`create_pyramid`][topozarr.coarsen.create_pyramid] is lazy — no data
is written. It produces a [`Pyramid`][topozarr.pyramid.Pyramid]
holding:

- **level_templates**: per-level `xr.Dataset`s
- **encoding**: chunk and shard sizes per variable per level.
- **attrs**: root metadata following the zarr-conventions
  [multiscales](https://github.com/zarr-conventions/multiscales),
  [proj](https://github.com/zarr-conventions/geo-proj), and
  [spatial](https://github.com/zarr-conventions/spatial) specs.

There are two ways to materialize the plan:

- **`Pyramid.write`** (default): level 0 is streamed from the source dataset,
  then each level `N` is block-reduced from the already-written level `N - 1`
  through the Rust kernel (`topozarr_core.block_reduce`), so the source is read
  exactly once regardless of the number of levels. Work runs on a local thread
  pool (not Dask). The rest of this document describes this path.
- **`Pyramid.as_datatree`**: returns a lazy `xr.DataTree` (levels coarsened via
  `xarray.coarsen`) for Dask-distributed writes. You call `to_zarr` yourself,
  passing `pyramid.encoding`.

## Chunk and shard heuristics

Spatial dimensions aim for chunks of `target_chunk_bytes` (default
~500 KB, sized for web visualization).

Shards group `chunks_per_shard` chunks per spatial dimension (default 4, i.e.
4×4 = 16 chunks, ~8 MB). Shards are also the unit of work during generation:
larger shards mean fewer, bigger reads/writes and more memory per worker.

`chunks_per_shard` sets a shard *byte budget* as well. A spatial dimension can
only hold as many chunks as fit whole, so a small raster — or any sufficiently
coarse pyramid level — leaves part of that budget unspent. The remainder widens
non-spatial dimensions instead of being discarded, innermost first, bounded by
each dimension's extent.

## Kernel semantics

`topozarr_core.block_reduce`

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
