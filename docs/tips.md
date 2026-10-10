# Tips & caveats

## Input requirements

`create_pyramid` validates these when the plan is built, so a bad input should fail
before anything is written:

- **`method`** must be one of `mean`, `max`, `min`, `sum`, `nearest` — checked
  against `topozarr_core.METHODS`.
- **Spatial coordinates must be 1-D** and uniformly spaced. Curvilinear grids
  (a 2-D `lat(y, x)` / `lon(y, x)`) are rejected.
- **Spatial variables** are limited to 4 dimensions.
- **Packed integers** (`scale_factor` / `add_offset`, e.g. Sentinel-2 `uint16`)
  are written packed whether the source was opened decoded or with
  `mask_and_scale=False`. Decoded input is re-packed to its stored dtype.
  Packed input without a `_FillValue`, `_Unsigned` variables, and masked-only
  integers (e.g. WorldCover `uint8`) stay float; open them with
  `mask_and_scale=False` to keep the integer dtype.

Need a different grid? Reproject or regrid upstream, then hand the result to `create_pyramid`.

## Faster local writes

With the `zarrs` extra installed, `write` uses the Rust [zarrs](https://github.com/zarrs/zarrs-python) codec pipeline automatically for local stores. Remote stores keep zarr-python's pipeline until [zarrs-python#139](https://github.com/zarrs/zarrs-python/issues/139) lands. To opt in anyway, set it yourself; `write` leaves an explicitly configured pipeline alone:

```python
import zarr
import zarrs

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

## Progress and memory

Pass `progress=True` to show a [tqdm](https://tqdm.github.io/) bar over written regions (requires `tqdm` to be installed):

```python
pyramid.write("pyramid.zarr", progress=True)
```

The thread pool size is auto-derived from CPU count and available RAM. Pass `max_workers` to override, and lower `max_region_bytes` (default 256 MB) to shrink level-0 tiles; peak memory is roughly `max_workers * 5 * max_region_bytes`.

`write` reads the source once, in level-0 tiles that cover whole shards of the finer levels, and writes each tile to every level it covers. Memory stays bounded by the tile size, not the raster size.

## Obstore timeouts

`obstore`'s defaults (5s connect / 30s total) can time out under heavy concurrency, surfacing as `GenericError` with `"Connect, TimedOut"`. Raise them via `client_options`, and consider raising `zarr.config`'s `async.concurrency` for higher S3 throughput:

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

If connect timeouts persist on large instances, try reducing `async.concurrency` or passing a smaller `max_workers`.

## Dask distributed

`Pyramid.write` does not use Dask — it streams regions through a local thread pool. For Dask-distributed writes, use `as_datatree()`, which returns a lazy `xr.DataTree` with all levels coarsened via `xarray.coarsen`. Pass `pyramid.encoding` to `to_zarr` to keep the recommended chunking and sharding:

```python
dt = pyramid.as_datatree()
dt.to_zarr("pyramid.zarr", zarr_format=3, consolidated=False,
           encoding=pyramid.encoding)
```

Deep levels of a Dask-backed source can outrun the chunk band that
`recommend_encoding` flexes to; if `to_zarr` raises on `safe_chunks`, pass
`safe_chunks=False`.

## Coming from ndpyramid

[ndpyramid](https://github.com/carbonplan/ndpyramid) was built for [carbonplan-maps](https://github.com/carbonplan/maps), which requires EPSG:3857, square, slippy-map-tile-compliant shapes. [zarr-layer](https://zarr-layer.demo.carbonplan.org/) relaxes these requirements, so topozarr only coarsens and writes metadata.

| ndpyramid | topozarr |
| --- | --- |
| `pyramid_coarsen` | `create_pyramid(ds, levels=...)` |
| `pyramid_reproject` | no equivalent — reproject upstream, then `create_pyramid` |
| `pyramid_regrid` | no equivalent — regrid upstream, then `create_pyramid` |
