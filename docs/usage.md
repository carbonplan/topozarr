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
    method="mean",  # "mean" (default) | "max" | "min" | "sum"
)

# compute and write all levels
pyramid.write("pyramid.zarr")
```

`levels` is the total number of resolution levels including the original. Level `0` is the original (highest) resolution; each subsequent level is coarsened by 2× per spatial dimension.

To build a sparse or non-uniform pyramid, pass `factors` instead of `levels` — explicit cumulative downsample factors per level, e.g. `factors=[1, 4, 16]` for native, 4×, and 16×. The list must start at `1`, be strictly increasing, and have each entry integer-divide the next. `levels=N` is equivalent to `factors=[1, 2, ..., 2**(N-1)]`.

```python
pyramid = create_pyramid(ds, factors=[1, 4, 16])
```

Levels are always named sequentially (`0, 1, 2, …`) regardless of `factors`; the downsample factor isn't in the node name but in the multiscales metadata (`layout[i].transform.scale` and each level's `spatial:transform`).

## Metadata only (no pyramid)

Low-resolution datasets don't need a pyramid. `attach_geozarr_metadata` returns
the dataset with the geozarr convention attrs (`proj:*`, `spatial:*`,
`zarr_conventions`) attached — no coarsening, no `/0` nesting, no `multiscales`
attr. Write it as a flat zarr group yourself:

```python
from topozarr import attach_geozarr_metadata

ds = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
ds.to_zarr("flat.zarr", zarr_format=3, consolidated=False)
```

CRS is read from the dataset (xproj) or passed explicitly via `crs="EPSG:4326"`.
Visualization hints work the same as `create_pyramid` via `layer_hints`.

## Dask distributed

`write()` is **not** Dask — it streams regions through a local thread pool. For Dask-distributed writes, use `as_datatree()`, which returns a lazy `xr.DataTree` with all levels coarsened via `xarray.coarsen`. The recommended per-level chunking and sharding lives in `pyramid.encoding` (already shaped for `DataTree.to_zarr`) — don't forget to pass it!

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

The threadpool size is auto-derived from CPU count and available RAM. Pass `max_workers` to override, and lower `max_region_bytes` (default 256 MB) to shrink level-0 read regions on chunked sources. For bounded memory on large stores, open the source lazily (e.g. `xr.open_zarr(store, chunks=None)`). See [Design](design.md#streaming-memory-model).

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

The heuristics target ~500 KB chunks for web visualization. Tune shard size with `chunks_per_shard` — chunks per shard along each spatial dimension (default `4`). Valid values are powers of 2: `1, 2, 4, 8, 16, 32`. Larger shards mean fewer, bigger reads/writes and higher memory (shards are the unit of work — see [Design](design.md#chunk-and-shard-heuristics)).

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
    from_url(
        "s3://carbonplan-scratch/topozarr/air.zarr",
        region="us-west-2",
        # defaults (5s connect / 30s total) can time out under heavy
        # concurrency; symptom: GenericError with "Connect, TimedOut"
        client_options={"connect_timeout": "30s", "timeout": "120s"},
    )
)
# raise async concurrency for higher S3 throughput
zarr.config.set({"async.concurrency": 128})
pyramid.write(store, mode="w")
```

If connect timeouts persist on large instances, lower the request fan-out (total in-flight requests is roughly `max_workers * async.concurrency`): reduce `async.concurrency` or pass a smaller `max_workers`.

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

## Experimental: Rust write path

Passing `io="rust"` writes through Rust using the Zarrs crate instead of `zarr-python` (no extra install):

```python
pyramid.write("s3://bucket/pyramid.zarr", io="rust")
```

Often faster on object stores (~25% on S3 in our benchmarks). Supports local paths, `s3://` URLs, `LocalStore`, and obstore-backed `ObjectStore` targets.

**Experimental:** the API may change.
