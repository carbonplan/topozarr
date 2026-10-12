# How it works

This page describes what `topozarr` does under the hood: how it plans a pyramid, how it writes one, and how each coarsening method combines cells.

## Plan, then write

[`create_pyramid`][topozarr.coarsen.create_pyramid] is lazy. It returns a
[`Pyramid`][topozarr.pyramid.Pyramid] plan and writes nothing.

<div class="tz-figure">
--8<-- "docs/assets/figures/plan-write.svg"
</div>

| Field | Holds |
|-------|-------|
| `level_templates` | One `xr.Dataset` per level |
| `encoding` | Chunk and shard sizes for each variable at each level |
| `attrs` | Root metadata: [multiscales](https://github.com/zarr-conventions/multiscales), [proj](https://github.com/zarr-conventions/proj), [spatial](https://github.com/zarr-conventions/spatial) ([layout](geozarr.md)) |

There are two ways to turn the plan into data:

- **`Pyramid.write`** (default) uses a local thread pool and reads the source
  only once.
- **`Pyramid.as_datatree`** returns a lazy `xr.DataTree` built with
  `xarray.coarsen`, for use with Dask. You then call `to_zarr` with
  `pyramid.encoding`.

## Write path

`write` splits the source into level `0` tiles, each covering whole shards of
levels `0..k`. Each worker:

1. Writes its tile to level `0`.
2. For each level `1..k`, reduces the previous level's block with the Rust
   kernel (`topozarr_core.block_reduce`), then writes the result.

This way, levels `0..k` never re-read from the store or share buffers between
workers.

<div class="tz-figure">
--8<-- "docs/assets/figures/write-path.svg"
</div>

??? note "How `k` is chosen"

    `k` is the deepest level whose tile still fits in `max_region_bytes` and
    leaves at least one tile per worker. Each level after `k` is reduced
    from the previous level, which is already written.

## Coarsening methods

<div class="tz-figure">
--8<-- "docs/assets/figures/methods.svg"
</div>

| Method | Each output cell is | Use for |
|--------|---------------------|---------|
| `mean` (default) | Window mean. Integers round half to even | Continuous data |
| `max`, `min` | Window maximum or minimum | Peaks and extents |
| `sum` | Window sum | Counts and totals |
| `nearest` | Top-left cell of the window | Categorical data (class codes, masks) |

- **Missing values:** NaN and `_FillValue` are skipped. A window with only
  missing values gives 0 for `sum` and the fill value (or NaN) for the other
  methods. `nearest` ignores both and copies the cell as is.
- **Dtypes:** `u8`, `u16`, `i16`, `i32`, `i64`, `f32`, and `f64`. Integers
  stay integers (unlike `xarray.coarsen`, which promotes them to float).
- **Shape:** matches `xarray.coarsen(boundary="trim")`, so partial windows at
  the trailing edges are dropped.

<div class="tz-figure">
--8<-- "docs/assets/figures/trim.svg"
</div>
