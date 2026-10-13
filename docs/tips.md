# Tips and caveats

This page collects input requirements, performance settings, and a few problems you may run into on large or remote writes.

## Input requirements

`create_pyramid` checks these when it builds the plan, so a bad input should fail
before anything is written:

- **`method`** must be one of `mean`, `max`, `min`, `sum`, `nearest`.
- **Spatial coordinates must be 1-D** and uniformly spaced. Curvilinear grids
  (a 2-D `lat(y, x)` / `lon(y, x)`) are rejected, because each level is described
  by [one affine transform](geozarr.md#spatial).
- **Spatial variables** can have at most four dimensions.
- **Scaled integers** (stored as integers with a `scale_factor` / `add_offset`)
  are written as integers, even if you opened them as floats.

If you need a different grid, reproject or regrid first, then pass the result to `create_pyramid` (see the [regridding example](examples/regrid.md)).

??? note "Scaled and masked integer details"

    CF calls these packed variables. Xarray decodes them to floats on open,
    which can make them 4x larger, so `create_pyramid` converts them back to
    their stored integer dtype. Opening them with `mask_and_scale=False` keeps
    them as integers from the start.

    Three cases are written as floats: scaled integers without a `_FillValue`,
    `_Unsigned` variables, and integers that only have a `_FillValue` (for
    example, WorldCover `uint8`). Open these with `mask_and_scale=False` to
    keep the integer dtype.

## Progress and memory

Pass `progress=True` to show a [tqdm](https://tqdm.github.io/) progress bar as regions are written (requires `tqdm`):

```python
pyramid.write("pyramid.zarr", progress=True)
```

`write` sizes its thread pool from the CPU count and available memory. Pass `max_workers` to override it. To shrink the level `0` tiles, lower `max_region_bytes` (default 256 MB). Peak memory is roughly `max_workers * 5 * max_region_bytes`.

See [How it works](design.md#write-path) for how tiles are chosen.

## Dask distributed

`Pyramid.write` uses a local thread pool, not Dask. To write with a Dask cluster, call `pyramid.as_datatree()` and pass `pyramid.encoding` to `to_zarr` (see the [Dask example](examples/dask.md)). If `to_zarr` raises a `safe_chunks` error, pass `align_chunks=True`.

## Faster local writes

With the `zarrs` extra installed, `write` automatically uses the Rust [zarrs](https://github.com/zarrs/zarrs-python) codec pipeline for local stores. Remote stores keep the zarr-python pipeline until [zarrs-python#139](https://github.com/zarrs/zarrs-python/issues/139) is resolved. To use `zarrs` for remote stores anyway, set it yourself. `write` leaves an explicitly configured pipeline alone:

```python
import zarr
import zarrs

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

## Obstore timeouts

The default `obstore` timeouts (5 s to connect, 30 s total) can expire under heavy concurrency. The error shows up as a `GenericError` containing `"Connect, TimedOut"`. Raise the timeouts with `client_options`. For higher S3 throughput, you can also raise `async.concurrency` in `zarr.config`:

```python
store = ObjectStore(
    from_url(
        "s3://carbonplan-scratch/topozarr/air.zarr",
        region="us-west-2",
        client_options={"connect_timeout": "30s", "timeout": "120s"},
    )
)
zarr.config.set({"async.concurrency": 128})
```

If connect timeouts continue on large instances, try lowering `async.concurrency` or passing a smaller `max_workers`.

## Coming from ndpyramid

We built [`ndpyramid`](https://github.com/carbonplan/ndpyramid) for [`@carbonplan/maps`](https://github.com/carbonplan/maps), which requires data in Web Mercator (EPSG:3857) with square chunks that match slippy map tiles. [`zarr-layer`](https://zarr-layer.demo.carbonplan.org/) drops these requirements, so `topozarr` only coarsens the data and writes metadata.

| ndpyramid | topozarr |
| --- | --- |
| `pyramid_coarsen` | `create_pyramid(ds, levels=...)` |
| `pyramid_reproject` | No equivalent. Reproject first, then call `create_pyramid` |
| `pyramid_regrid` | No equivalent. Regrid first, then call `create_pyramid` |
