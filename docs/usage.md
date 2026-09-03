# Usage

## Basic example

Load an Xarray dataset, create a pyramid, then write it:

```python
import xarray as xr
import xproj  # for CRS assignment
from topozarr import create_pyramid

ds = xr.tutorial.open_dataset('air_temperature').drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    method="mean",  # "mean" (default) | "max" | "min" | "sum" | "nearest"
)

# compute and write all levels
pyramid.write("pyramid.zarr")
```

`levels` is the total number of resolution levels including the original. Level `0` is the original (highest) resolution; each subsequent level is coarsened by 2× per spatial dimension.

To build a non-uniform pyramid, pass `factors` instead of `levels` — explicit cumulative downsample factors per level, e.g. `factors=[1, 4, 16]` for native, 4×, and 16×.

```python
pyramid = create_pyramid(ds, factors=[1, 4, 16])
```

Levels are always named sequentially (`0, 1, 2, …`) regardless of if you specify `factors`; the downsample factor isn't in the node name but in the multiscales metadata (`layout[i].transform.scale` and each level's `spatial:transform`).

## Input requirements

`create_pyramid` validates these when the plan is built, so a bad input should fail
before anything is written:

- **`method`** must be one of `mean`, `max`, `min`, `sum`, `nearest` — checked
  against `topozarr_core.METHODS`.
- **Spatial coordinates must be 1-D** and uniformly spaced. Curvilinear grids
  (a 2-D `lat(y, x)` / `lon(y, x)`) are rejected.
- **Spatial variables** are limited to 4 dimensions.

## Single-resolution datasets (no pyramid)

Lower-resolution datasets often don't need overviews to be visualized with `zarr-layer`. `topozarr` provides two functions that can help visualize Zarr stores without overviews.
`attach_geozarr_metadata` returns the dataset with the geozarr convention attrs
(`proj:*`, `spatial:*`, `zarr_conventions`) attached and `recommend_encoding` returns the same
chunk/shard heuristic `create_pyramid` applies per level.
`zarr-layer` can render stores without overviews as long as the chunking is friendly for web mapping, so this is a real option and not just a
data-production convenience. Just remember that there are no overviews, so a zoomed-out read
still pulls the full resolution.

```python
from topozarr import attach_geozarr_metadata, recommend_encoding

ds = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
ds.to_zarr(
    "flat.zarr",
    zarr_format=3,
    consolidated=False,
    encoding=recommend_encoding(ds, x_dim="lon", y_dim="lat"),
)
```

## Dask distributed

Pyramid `write()` does not use Dask — it streams regions through a local thread pool. For Dask-distributed writes, use `as_datatree()`, which returns a lazy `xr.DataTree` with all levels coarsened via `xarray.coarsen`. The recommended per-level chunking and sharding lives in `pyramid.encoding` (already shaped for `DataTree.to_zarr`) — don't forget to pass the recommended encoding in your `to_zarr(..., encoding=pyramid.encoding)` call.

```python
dt = pyramid.as_datatree()
dt.to_zarr("pyramid.zarr", zarr_format=3, consolidated=False,
           encoding=pyramid.encoding)
```

## Progress and memory

Pass `progress=True` to show a [tqdm](https://tqdm.github.io/) bar over written regions (requires `tqdm` to be installed):

```python
pyramid.write("pyramid.zarr", progress=True)
```

The threadpool size is auto-derived from CPU count and available RAM. Pass `max_workers` to override, and lower `max_region_bytes` (default 256 MB) to shrink level-0 read regions on chunked sources.

Pass `keep_levels_in_memory=True` to keep levels in RAM and skip re-reading them from the store between levels (faster, but uses more memory). `None` (default) enables this automatically when subsequent levels fit in RAM.

## Visualization hints

Optional. If you'll render the pyramid in [zarr-layer](https://zarr-layer.demo.carbonplan.org/), `layer_hints` embeds a default colormap and color range so it displays sensibly without manual setup. Skip it otherwise — it has no effect on the data.

```python
from topozarr.metadata import ZarrLayerVarConfig

pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    layer_hints={"air": ZarrLayerVarConfig(colormap="blues", clim=[230, 310])},
)
```

Written into the root `zarr-layer` metadata key; nothing else changes.

## Chunking

`pyramid.encoding` holds the chunk and shard sizes per variable per level; `pyramid.write` applies them automatically.

The heuristics target ~500 KB spatial chunks for web visualization. Tune shard size with `chunks_per_shard` — chunks per shard along each spatial dimension (default `4`). Valid values are powers of 2: `1, 2, 4, 8, 16, 32`. Larger shards mean fewer, bigger reads/writes and higher memory (shards are the unit of work — see [Design](design.md#chunk-and-shard-heuristics)).

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|:------------:|:-----------------:|
| 1 | 1 | ~500 KB |
| 4 (default) | 16 | ~8 MB |
| 8 | 64 | ~32 MB |
| 16 | 256 | ~128 MB |

Pass `chunks_per_shard=None` to disable sharding entirely.

### Non-spatial dimensions

`chunks_per_shard` also sets a shard byte budget. Spatial dimensions are sized first; whatever is left over widens non-spatial dimensions (`time`, `band`, ...) instead of leaving them at one element per shard. Chunk size along those dimensions stays 1, so reads still fetch a single element.

To override, edit `pyramid.encoding` before writing. Chunk and shard values are plain tuples in dimension order, so use `.dims` to find the axis — it differs between variables:

```python
enc = pyramid.encoding["/0"]["wind_speed"]
axis = pyramid.level_templates[0]["wind_speed"].dims.index("time")

shards = list(enc["shards"])
shards[axis] = 1  # one timestep per shard
enc["shards"] = tuple(shards)
```

Repeat per level and variable. Zarr requires each shard to be a whole multiple of its chunk.

## Writing backends

`pyramid.write` accepts a local path, an `Obstore` store, or an `Icechunk` store.

### Local path

```python
pyramid.write("pyramid.zarr")
```

### Icechunk

```python
import icechunk

storage = icechunk.s3_storage(
    bucket="<your_bucket>", prefix="<your_prefix>", from_env=True
)
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")
pyramid.write(session.store, mode="w")
session.commit("write pyramid")
```

### Obstore

```python
from obstore.store import from_url
from zarr.storage import ObjectStore

store = ObjectStore(from_url("s3://carbonplan-scratch/topozarr/air.zarr", region="us-west-2"))
pyramid.write(store, mode="w")
```

**Tuning / troubleshooting:** `obstore`'s defaults (5s connect / 30s total) can time out under
heavy concurrency, surfacing as `GenericError` with `"Connect, TimedOut"`. Raise them via
`client_options`, and consider raising `zarr.config`'s `async.concurrency` for higher S3
throughput:

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

If connect timeouts persist on large instances, try reducing `async.concurrency` or passing a
smaller `max_workers`.

