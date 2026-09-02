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

To build a sparse or non-uniform pyramid, pass `factors` instead of `levels` — explicit cumulative downsample factors per level, e.g. `factors=[1, 4, 16]` for native, 4×, and 16×. The list must start at `1`, be strictly increasing, and have each entry integer-divide the next. `levels=N` is equivalent to `factors=[1, 2, ..., 2**(N-1)]`.

```python
pyramid = create_pyramid(ds, factors=[1, 4, 16])
```

Levels are always named sequentially (`0, 1, 2, …`) regardless of `factors`; the downsample factor isn't in the node name but in the multiscales metadata (`layout[i].transform.scale` and each level's `spatial:transform`).

## Input requirements

`create_pyramid` validates these when the plan is built, so a bad input fails
before anything is written:

- **`method`** must be one of `mean`, `max`, `min`, `sum`, `nearest` — checked
  against `topozarr_core.METHODS`, the list the installed kernel actually
  implements.
- **Spatial coordinates must be 1-D** and uniformly spaced. Curvilinear grids
  (a 2-D `lat(y, x)` / `lon(y, x)`) are rejected: topozarr coarsens 1-D
  coordinates only, so 2-D ones would be left at native resolution and
  mis-register the coarsened levels. Reproject to a regular grid first, or
  drop them with `ds.drop_vars(["lat", "lon"])` if they are redundant.
- **Spatial variables** are limited to 4 dimensions (the kernel's limit). Use
  `as_datatree()` for the xarray/Dask path, which lifts it.
- **Non-numeric variables over a spatial dim** (string labels, datetimes) are
  rejected: neither path can reduce them. Drop them with `ds.drop_vars([...])`.

A variable over only *one* spatial dim — a per-column `profile(time, x)`, say —
is coarsened along that dim alone, on both `write` and `as_datatree`. Variables
and coordinates over neither spatial dimension are untouched, 2-D or not.

## Single-resolution datasets (no pyramid)

Low-resolution datasets don't need a pyramid. Two functions cover the flat path:
`attach_geozarr_metadata` returns the dataset with the geozarr convention attrs
(`proj:*`, `spatial:*`, `zarr_conventions`) attached — no coarsening, no `/0`
nesting, no `multiscales` attr — and `recommend_encoding` returns the same
chunk/shard heuristic `create_pyramid` applies per level. Write it as a flat
zarr group yourself:

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

zarr-layer renders a flat group, so this is a real option and not just a
data-production convenience — but there are no overviews, so a zoomed-out read
still pulls full resolution.

`create_pyramid(levels=1)` is **not** the way to get a flat dataset: it still
produces `/0` nesting and a one-entry `multiscales` attr.

CRS is read from the dataset (xproj) or passed explicitly via `crs="EPSG:4326"`.
Visualization hints work the same as `create_pyramid` via `layer_hints`.
`recommend_encoding` needs no CRS — the encoding depends only on shape and
dtype. It covers variables over at least one spatial dim — one with only a
single spatial dim is sized along that dim alone; anything over neither falls
through to xarray's defaults.

For a **dask-backed** dataset, xarray's `safe_chunks` check requires the zarr
write unit — the *shard*, when sharding is on — to divide the dask block.
`recommend_encoding` treats `chunks_per_shard` as an upper bound and flexes it
down until a shard does, so the snippet above writes a dask source as is.

The exception is a source chunk too small to divide into a chunk of usable size
(under ~128 elements, or under half the ideal chunk). The recommendation keeps a
read-aligned shard there, and such a write still needs a rechunk or
`safe_chunks=False`:

```python
enc = recommend_encoding(ds)["elevation"]
ds = ds.chunk(dict(zip(ds.elevation.dims, enc["shards"])))
```

A lazily opened zarr-backed dataset (`xr.open_dataset(..., chunks=None)`) is
unaffected — nothing checks it.

## Dask distributed

`write()` is **not** Dask — it streams regions through a local thread pool. For Dask-distributed writes, use `as_datatree()`, which returns a lazy `xr.DataTree` with all levels coarsened via `xarray.coarsen`. The recommended per-level chunking and sharding lives in `pyramid.encoding` (already shaped for `DataTree.to_zarr`) — don't forget to pass it!

```python
dt = pyramid.as_datatree()
dt.to_zarr("pyramid.zarr", zarr_format=3, consolidated=False,
           encoding=pyramid.encoding)
```

This path produces the same values and dtypes as `write()`, `_FillValue`
handling included. Each coarsen runs on an `f8` promotion to match the kernel's
accumulator, so a `u1` variable is momentarily 8x its stored size — bounded by
the dask block, not the array. An `f8` source is the one case the two paths
differ, by under 1 ULP on `mean`/`sum`.

`pyramid.encoding` is shard-aligned to the source chunking at every level, not
just level 0, so this writes without `safe_chunks=False` as long as the
coarsened dask blocks stay large enough to divide into a usable chunk — roughly
half the ideal chunk size, ~181 elements for `f4` at the default target. Past
that depth the levels fall out of alignment and need `safe_chunks=False`. A
2000² `f4` source chunked at 1000 stays aligned for three levels; chunked at
500 it aligns for two.

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
