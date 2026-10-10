# How it works

## Plan, then write

[`create_pyramid`][topozarr.coarsen.create_pyramid] is lazy: it returns a
[`Pyramid`][topozarr.pyramid.Pyramid] plan and writes nothing.

| Field | Holds |
|-------|-------|
| `level_templates` | per-level `xr.Dataset`s |
| `encoding` | chunk and shard sizes per variable per level |
| `attrs` | root metadata: [multiscales](https://github.com/zarr-conventions/multiscales), [proj](https://github.com/zarr-conventions/proj), [spatial](https://github.com/zarr-conventions/spatial) |

Two ways to materialize it:

- **`Pyramid.write`** (default): local thread pool, reads the source once.
- **`Pyramid.as_datatree`**: lazy `xr.DataTree` via `xarray.coarsen`, for
  Dask. You call `to_zarr` with `pyramid.encoding`.

## Write path

The source is split into level-0 tiles, each covering whole shards of levels
`0..k`. Each worker:

1. Writes its tile to level 0.
2. Reduces it with the Rust kernel (`topozarr_core.block_reduce`).
3. Writes the result to levels `1..k`.

No store re-reads, no shared buffers.

??? note "How `k` is chosen"

    `k` is the deepest level whose tile still fits `max_region_bytes` and
    leaves at least one tile per worker. Levels above `k` are block-reduced
    from the already-written level `N - 1`.

## Coarsening methods

| Method | Each output cell is | Use for |
|--------|---------------------|---------|
| `mean` (default) | window mean; integers round half to even | continuous data |
| `max` / `min` | window max / min | peaks, extents |
| `sum` | window sum | counts, totals |
| `nearest` | top-left cell of the window | categorical data (class codes, masks) |

- **Missing values:** NaN and `_FillValue` are skipped. An all-missing window
  gives 0 for `sum` and the fill value (or NaN) otherwise. `nearest` ignores
  both.
- **Dtypes:** `u8`, `u16`, `i16`, `i32`, `i64`, `f32`, `f64`; integers stay
  integer (unlike `xarray.coarsen`, which promotes to float).
- **Shape:** matches `xarray.coarsen(boundary="trim")`; trailing partial
  windows are dropped.
