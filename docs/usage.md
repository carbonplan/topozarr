# Usage

## Basic example

Load a georeferenced dataset, create a pyramid plan, then write it:

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
    method="mean",  # "mean" (default) | "max" | "min" | "sum"
)

# inspect chunk/shard encoding before writing
print(pyramid.encoding)

# compute and write all levels
pyramid.write("pyramid.zarr")
```

`levels` is the total number of resolution levels including the original. Level `0` is the original (highest) resolution; each subsequent level is coarsened by 2× per spatial dimension.

`create_pyramid` returns a write plan. `pyramid.write(store)` does the work: level 0 is copied from the source dataset, then each level is block-reduced from the previously written one, streaming shard-sized regions through the Rust kernel on a thread pool. Tune parallelism with `max_workers`. Reduction semantics match `xarray.coarsen(boundary="trim")` exactly, including NaN / `_FillValue` handling.

## Visualization hints

Embed colormap and color-range hints for [zarr-layer](https://zarr-layer.demo.carbonplan.org/) directly in the pyramid metadata:

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

These are written into the root `zarr-layer` metadata key. Omitting `layer_hints` has no effect on the pyramid structure or encoding.

## Chunking

`pyramid.encoding` holds the chunk and shard sizes per variable per level; `pyramid.write` applies them automatically.

The heuristics target ~500 KB chunks for web visualization. Tune shard size with `chunks_per_shard` — the number of chunks per shard along each spatial dimension (default: `4`, giving 4×4 = 16 chunks per shard and ~8 MB shards). Valid values are powers of 2: `1, 2, 4, 8, 16, 32`. Shards are also the unit of work during pyramid generation, so larger shards mean fewer, bigger reads/writes and higher memory usage.

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|:------------:|:-----------------:|
| 1 | 1 | ~500 KB |
| 4 (default) | 16 | ~8 MB |
| 8 | 64 | ~32 MB |
| 16 | 256 | ~128 MB |

Pass `chunks_per_shard=None` to disable sharding entirely.

## Writing backends

`pyramid.write` accepts anything `zarr-python` can open — a local path, an `ObjectStore`, or an icechunk session store.

### Object storage

```python
from obstore.store import from_url
from zarr.storage import ObjectStore

store = ObjectStore(
    from_url("s3://carbonplan-scratch/topozarr/air.zarr", region="us-west-2")
)
pyramid.write(store, mode="w")
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

## Optional: zarrs codec pipeline

Compression codec work can optionally be routed through the Rust [zarrs](https://github.com/zarrs/zarrs-python) codec pipeline. It plugs in at the codec layer and works with any store backend:

```bash
uv add zarrs
```

```python
import zarr
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```
