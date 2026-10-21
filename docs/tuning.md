# Tuning

The defaults work well for most data, so you may not need this page. When you do, these are the settings that matter. The examples use the `air_temperature` tutorial dataset.

| Setting | Where | Effect |
|------|-------|--------|
| [`levels` or `factors`](#level-counts) | `create_pyramid` | Number of levels, or explicit downsampling factors |
| [`target_chunk_bytes`](#chunks-and-shards) | `create_pyramid` | Chunk size on disk (default ~500 KB) |
| [`chunks_per_shard`](#chunks-and-shards) | `create_pyramid` | Shard size, which is also the unit of work. `None` disables sharding |
| [`progress`, `max_workers`](tips.md#progress-and-memory) | `Pyramid.write` | Progress bar, thread count, and memory |

```python exec="on" source="above" session="tuning" result="python"
import xarray as xr
import xproj  # noqa: F401

from topozarr import create_pyramid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
print(dict(ds.sizes))
```

## Level counts

`levels="auto"` keeps adding 2x coarser levels until the smaller spatial dimension would drop below `min_dim`. The default of 256 is about one web map tile. We lower it here because the tutorial grid is small:

```python exec="on" source="above" session="tuning" result="python"
pyramid = create_pyramid(ds, levels="auto", min_dim=6, x_dim="lon", y_dim="lat")
for level, template in pyramid.level_templates.items():
    print(level, dict(template.sizes))
```

`factors` sets explicit downsampling factors instead, each relative to level `0` (here, native resolution and 4x). Levels are still named `0, 1, ...`, and each level's factor is recorded in the multiscales metadata:

```python exec="on" source="above" session="tuning" result="python"
sparse = create_pyramid(ds, factors=[1, 4], x_dim="lon", y_dim="lat")
print(dict(sparse.level_templates[1].sizes))
```

## Chunks and shards

`pyramid.encoding` holds the chunk and shard sizes for each variable at each level, and `write` applies them. Spatial chunks are sized to approach `target_chunk_bytes`. `chunks_per_shard` (1, 2, 4, 8, 16, 32, or `None`) sets the number of chunks per shard along each spatial dimension. Shards are the unit of work, so bigger shards mean fewer, larger writes and more memory per worker.

<div class="tz-figure">
--8<-- "docs/assets/figures/chunks-shards.svg"
</div>

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|:------------:|:-----------------:|
| `None` | no sharding | — |
| 1 | 1 | ~500 KB |
| 4 (default) | 16 | ~8 MB |
| 8 | 64 | ~32 MB |
| 16 | 256 | ~128 MB |

On small grids, any leftover shard budget goes to non-spatial dimensions like `time`. Chunks along those dimensions stay at size 1:

<div class="tz-figure">
--8<-- "docs/assets/figures/time-shards.svg"
</div>

```python exec="on" source="above" session="tuning" result="python"
for cps in [1, 4, None]:
    enc = create_pyramid(ds, levels=2, chunks_per_shard=cps, x_dim="lon", y_dim="lat").encoding
    print(f"chunks_per_shard={cps}: {enc['/0']['air']}")
```

??? note "Custom chunks and shards"

    Edit `pyramid.encoding` before writing. Values are tuples in dimension
    order (`time, lat, lon`), and each shard must be a whole multiple of its
    chunk. This example puts 365 time steps in each shard on every level:

    ```python exec="on" source="above" session="tuning" result="python"
    for level_encoding in pyramid.encoding.values():
        _, *spatial = level_encoding["air"]["shards"]
        level_encoding["air"]["shards"] = (365, *spatial)
    print(pyramid.encoding["/0"]["air"])
    ```

??? note "Rewrite some levels and get per-level timings"

    `levels=[...]` with `mode="a"` rewrites a subset of levels and keeps the
    rest. Level `N - 1` must be in the subset or already in the store, and
    the pyramid must match the levels the store already lists (same method
    and shapes, at least as many levels).
    `stats=True` returns timings for each level.

    ```python
    pyramid.write("pyramid.zarr")
    stats = pyramid.write("pyramid.zarr", mode="a", levels=[1, 2], stats=True)
    print({level: s["wall_s"] for level, s in stats.items()})
    ```
