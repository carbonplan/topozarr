# Release notes

## Unreleased

### Fixed

- `create_pyramid` now validates `method` when the plan is built instead of
  letting the Rust kernel reject it on the first *coarsened* level — by which
  point level 0 was already written to the store. `Pyramid.write` re-checks,
  so editing `pyramid.method` after planning is caught too, and a rejected
  write leaves the store untouched.

  The valid set comes from the installed kernel (`topozarr_core.METHODS`, new
  in core 0.1.7), not from a second list on the Python side. That is the check
  [#26](https://github.com/carbonplan/topozarr/issues/26) called for: a
  `topozarr` advertising a method its linked `topozarr-core` does not
  implement now fails at plan time.

- `as_datatree()` now matches `write()` value for value. `xarray.coarsen` knows
  nothing of `_FillValue`, so the Dask path averaged the sentinel in as data —
  silently wrong for masked rasters, with nothing in the output to hint at it.
  It also promoted integers to float, which left `pyramid.encoding` (sized from
  the source itemsize) mis-sized for the data being written through it. Levels
  are now coarsened on an `f8` promotion with the fill masked to NaN, then
  refilled, clipped and cast back to the source dtype. The clip matters for an
  integer `sum`: the kernel saturates an out-of-range accumulator where a bare
  numpy cast wraps.

  An `f8` source is the one remaining divergence — under 1 ULP on `mean`/`sum`,
  from window summation order. A method with no `xarray.coarsen` equivalent now
  raises `NotImplementedError` instead of `AttributeError`.

- `create_pyramid` rejects datasets whose spatial coordinates are 2-D
  (curvilinear grids, e.g. `lat(y, x)`). They were never coarsened: the level
  templates left them at native resolution, which surfaced as an opaque
  xarray `conflicting sizes for dimension` error, and `as_datatree` silently
  corner-strided them into a mis-registered grid. The error now names the
  coordinates and points at `ds.drop_vars`.

### Added

- `recommend_encoding(ds, x_dim=..., y_dim=...)` returns the chunk/shard
  encoding for a single-resolution (flat) dataset — the same heuristic
  `create_pyramid` applies per level, previously reachable only as
  `pyramid.encoding`. Pair it with `attach_geozarr_metadata` and pass the
  result to `ds.to_zarr(..., encoding=...)`. No CRS required; it covers
  variables with both spatial dims and leaves the rest to xarray's defaults.
  `create_pyramid` now builds its per-level encoding through the same
  function, so pyramid output is unchanged.

  Note for dask-backed datasets: xarray's `safe_chunks` check compares dask
  blocks against the zarr write unit (the shard, when sharding is on), and the
  recommendation snaps chunks but not shards. Rechunk to the recommended shards
  before writing, or pass `safe_chunks=False`. A lazily opened zarr-backed
  dataset is unaffected.

## 0.1.5

- Chunking/sharding heuristics now take into account the non-spatial dimensions
  (`time`, `band`, ...). If there is available 'space', then shards can include
  more chunks-per-shard for non-spatial dims. You can still specify your own encoding by editing
  `pyramid.encoding` before writing.

- Removed the experimental `io="rust"` write path (`Pyramid.write(..., io="rust")`) and the
  `RustWriter` class in `topozarr-core`. It wrote regions through the `zarrs` crate
  instead of zarr-python for **roughly** a 25% gain on S3, which did not justify carrying a
  second write path, a store-URL translation layer, and four heavy Rust dependencies.
  All writes now go through zarr-python or Icechunk. `topozarr-core` continues to provide the
  `block_reduce` coarsening kernel, which is unaffected.

## 0.1.4

### Fixed

- `topozarr` now pins `topozarr-core` exactly instead of `>=0.1.0,<0.2`. The
  old range let a resolver pair topozarr 0.1.3 with core 0.1.0/0.1.1, which
  predate the `nearest` kernel: the Python layer accepted `method="nearest"`
  and the Rust kernel then raised `ValueError` mid-write, after a pyramid had
  started. Reported in [#26](https://github.com/carbonplan/topozarr/issues/26).

### Changed

- `topozarr` and `topozarr-core` are released in lockstep under the same
  version number from here on (core skips 0.1.3). A CI check keeps the pin and
  both core manifests in sync.

## 0.1.3

### Added

- `nearest` coarsening method, alongside `mean`/`max`/`min`/`sum`. Corner-picks
  the top-left cell of each window instead of aggregating, for categorical
  data (class codes, masks) where averaging invents values. Composable across
  levels like the other methods.

---

Versions prior to this point are not documented here — see the
[GitHub releases](https://github.com/carbonplan/topozarr/releases) and
[tags](https://github.com/carbonplan/topozarr/tags) for history.
