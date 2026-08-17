# Release notes

## 0.1.5

### Changed

- Chunking/sharding heuristics now take into account the non-spatial dimensions
  (`time`, `band`, ...). If there is available 'space', then shards can include
  more chunks-per-shard for non-spatial dims. You can still specify your own encoding by editing
  `pyramid.encoding` before writing.

- The experimental `io="rust"` write path (`Pyramid.write(..., io="rust")`) and the
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
