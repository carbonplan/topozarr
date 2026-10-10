# Tuning

## Quick reference

| Knob | Where | Effect |
|------|-------|--------|
| [`levels` / `factors`](usage.md#basic-example) | `create_pyramid` | number of levels, or explicit cumulative downsample factors |
| [`target_chunk_bytes`](#chunk-and-shard-sizes) | `create_pyramid` | chunk size on disk (default ~500 KB) |
| [`chunks_per_shard`](#chunk-and-shard-sizes) | `create_pyramid` | shard size = work unit; `None` disables sharding |
| [`max_region_bytes`](tips.md#progress-and-memory) | `Pyramid.write` | cap on level-0 tile size and per-worker memory |
| [`max_workers`](tips.md#progress-and-memory) | `Pyramid.write` | thread pool size; `None` = RAM/CPU-derived |
| [`progress`](tips.md#progress-and-memory) | `Pyramid.write` | tqdm bar over written regions |
| [codec pipeline](tips.md#faster-local-writes) | `zarr.config` | zarrs (Rust) used automatically for local stores with the `zarrs` extra |

## Chunk and shard sizes

`pyramid.encoding` holds the chunk and shard sizes per variable per level;
`pyramid.write` applies them automatically.

Spatial chunks target `target_chunk_bytes` (~500 KB, sized for web
visualization). `chunks_per_shard` sets chunks per shard along each spatial
dimension; valid values are `1, 2, 4, 8, 16, 32`. Shards are the unit of work
during writes: bigger shards mean fewer, larger reads/writes and more memory
per worker.

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|:------------:|:-----------------:|
| `None` | no sharding | — |
| 1 | 1 | ~500 KB |
| 4 (default) | 16 | ~8 MB |
| 8 | 64 | ~32 MB |
| 16 | 256 | ~128 MB |

## Non-spatial dimensions

`chunks_per_shard` also sets a shard byte budget. Any budget the spatial
dimensions can't use (small rasters, coarse levels) goes to non-spatial
dimensions (`time`, `band`, ...) instead. Chunk size along those dimensions stays 1,
so reads still fetch a single element.

??? note "Override shard shape per variable"

    Edit `pyramid.encoding` before writing. Values are tuples in dimension
    order, so use `.dims` to find the axis:

    ```python
    enc = pyramid.encoding["/0"]["wind_speed"]
    axis = pyramid.level_templates[0]["wind_speed"].dims.index("time")

    shards = list(enc["shards"])
    shards[axis] = 1  # one timestep per shard
    enc["shards"] = tuple(shards)
    ```

    Repeat per level and variable. Each shard must be a whole multiple of its
    chunk.
