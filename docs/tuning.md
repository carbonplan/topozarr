# Tuning

The defaults work for most data. Code blocks on this page run on the `air_temperature` tutorial dataset when the docs are built.

| Knob | Where | Effect |
|------|-------|--------|
| [`levels` / `factors`](#level-counts) | `create_pyramid` | number of levels, or explicit downsample factors |
| [`target_chunk_bytes`](#chunks-and-shards) | `create_pyramid` | chunk size on disk (default ~500 KB) |
| [`chunks_per_shard`](#chunks-and-shards) | `create_pyramid` | shard size = work unit; `None` disables sharding |
| [`max_workers`, `max_region_bytes`](tips.md#progress-and-memory) | `Pyramid.write` | threads and per-worker memory |
| [`progress`](tips.md#progress-and-memory) | `Pyramid.write` | tqdm bar over written regions |

```python exec="on" source="above" session="tuning" result="text"
import xarray as xr
import xproj  # noqa: F401

from topozarr import create_pyramid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
print(dict(ds.sizes))
```

## Level counts

`levels="auto"` adds 2x levels until the smaller spatial dim would drop below `min_dim` (default 256, about one web tile; lowered here for a small grid):

```python exec="on" source="above" session="tuning" result="text"
pyramid = create_pyramid(ds, levels="auto", min_dim=6, x_dim="lon", y_dim="lat")
for level, template in pyramid.level_templates.items():
    print(level, dict(template.sizes))
```

`factors` sets explicit cumulative downsample factors instead, e.g. native and 4x. Levels are still named `0, 1, ...`; the factor is in the multiscales metadata:

```python exec="on" source="above" session="tuning" result="text"
sparse = create_pyramid(ds, factors=[1, 4], x_dim="lon", y_dim="lat")
print(dict(sparse.level_templates[1].sizes))
```

## Chunks and shards

`pyramid.encoding` holds chunk and shard sizes per variable per level; `write` applies them. Spatial chunks target `target_chunk_bytes`. `chunks_per_shard` (1, 2, 4, 8, 16, 32, or `None`) sets chunks per shard along each spatial dim; shards are the unit of work, so bigger shards mean fewer, larger writes and more memory per worker.

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|:------------:|:-----------------:|
| `None` | no sharding | — |
| 1 | 1 | ~500 KB |
| 4 (default) | 16 | ~8 MB |
| 8 | 64 | ~32 MB |
| 16 | 256 | ~128 MB |

Shard budget the spatial dims can't use (small rasters, coarse levels) widens non-spatial dims like `time`, up to 128 chunks per shard. Chunks along those dims stay 1, so a read still fetches one timestep:

```python exec="on" source="above" session="tuning" result="text"
for cps in [1, 4, None]:
    enc = create_pyramid(ds, levels=2, chunks_per_shard=cps, x_dim="lon", y_dim="lat").encoding
    print(f"chunks_per_shard={cps}: {enc['/0']['air']}")
```

To override, edit `pyramid.encoding` before writing. Values are tuples in dimension order (`time, lat, lon`); here, 365 timesteps per shard on every level. Each shard must be a whole multiple of its chunk:

```python exec="on" source="above" session="tuning" result="text"
for level_encoding in pyramid.encoding.values():
    _, *spatial = level_encoding["air"]["shards"]
    level_encoding["air"]["shards"] = (365, *spatial)
print(pyramid.encoding["/0"]["air"])
```

??? note "Rewrite some levels, and per-level timings"

    `levels=[...]` with `mode="a"` rewrites a subset and keeps the rest; level
    `N - 1` must be in the subset or already in the store. `stats=True` returns
    per-level timings.

    ```python
    pyramid.write("pyramid.zarr")
    stats = pyramid.write("pyramid.zarr", mode="a", levels=[1, 2], stats=True)
    print({level: s["wall_s"] for level, s in stats.items()})
    ```
