# Usage

## Basic example

Load an Xarray dataset, create a pyramid, then write it:

```python
import xarray as xr
import xproj  # for CRS assignment
from topozarr import create_pyramid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
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

Levels are always named sequentially (`0, 1, 2, …`) regardless of whether you specify `factors`; the downsample factor isn't in the node name but in the multiscales metadata (`layout[i].transform.scale` and each level's `spatial:transform`).

## Writing backends

`pyramid.write` accepts a local path, an `Obstore` store, or an `Icechunk` store.

### Local path

```python
pyramid.write("pyramid.zarr")
```

For faster local writes, install the `zarrs` extra (see [Tips](tips.md#faster-local-writes)).

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

Seeing `"Connect, TimedOut"` errors? See [Tips](tips.md#obstore-timeouts).

## Single-resolution datasets (no pyramid)

Lower-resolution datasets often don't need overviews: `zarr-layer` can render a flat store as long as its chunking is web-friendly. `attach_geozarr_metadata` adds the geozarr convention attrs (`proj:*`, `spatial:*`, `zarr_conventions`), and `recommend_encoding` returns the same chunk/shard heuristic `create_pyramid` applies per level. Without overviews, a zoomed-out read still pulls the full resolution.

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

`pyramid.write` applies chunk and shard sizes from `pyramid.encoding` automatically. To tune them, see [Tuning](tuning.md#chunk-and-shard-sizes).
