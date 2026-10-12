# Tuning and flat stores

Level counts, chunk/shard sizes, encoding overrides, partial rewrites, and single-resolution stores, on the `air_temperature` tutorial dataset. See [Tuning](../tuning.md) for what each knob does.

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

`factors` sets explicit cumulative downsample factors instead; levels are still named `0, 1, ...`:

```python exec="on" source="above" session="tuning" result="text"
sparse = create_pyramid(ds, factors=[1, 4], x_dim="lon", y_dim="lat")
print(dict(sparse.level_templates[1].sizes))
```

## Chunks and shards

`chunks_per_shard` sets chunks per shard along each spatial dim (`None` disables sharding). A grid this small cannot fill the shard budget spatially, so the remainder widens `time`, capped at 128 inner chunks per shard. Chunks along `time` stay 1:

```python exec="on" source="above" session="tuning" result="text"
for cps in [1, 4, None]:
    enc = create_pyramid(ds, levels=2, chunks_per_shard=cps, x_dim="lon", y_dim="lat").encoding
    print(f"chunks_per_shard={cps}: {enc['/0']['air']}")
```

To override, edit `pyramid.encoding` before writing. Encodings are tuples in dimension order (`time, lat, lon`); here, 365 timesteps per shard on every level:

```python exec="on" source="above" session="tuning" result="text"
for level_encoding in pyramid.encoding.values():
    _, *spatial = level_encoding["air"]["shards"]
    level_encoding["air"]["shards"] = (365, *spatial)
print(pyramid.encoding["/0"]["air"])
```

## Write, then rewrite some levels

`stats=True` returns per-level timings. To rewrite a subset of levels and keep the rest, pass `levels` with `mode="a"`; level `N - 1` must be in the subset or already in the store:

```python exec="on" source="above" session="tuning" result="text"
stats = pyramid.write("tuning.zarr", stats=True)
print({level: s["wall_s"] for level, s in stats.items()})

pyramid.write("tuning.zarr", mode="a", levels=[1, 2])
print(xr.open_datatree("tuning.zarr", engine="zarr", consolidated=False).groups)
```

## Single-resolution stores

Small datasets often don't need overviews. `attach_geozarr_metadata` adds the convention attrs and `recommend_encoding` returns the chunk/shard heuristic `create_pyramid` uses per level, for a plain `to_zarr`:

```python exec="on" source="above" session="tuning" result="text"
from topozarr import attach_geozarr_metadata, recommend_encoding

flat = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
encoding = recommend_encoding(flat, x_dim="lon", y_dim="lat")
flat.to_zarr("flat.zarr", zarr_format=3, consolidated=False, encoding=encoding, mode="w")
print(encoding)
```

```python exec="on" session="tuning"
# hidden: remove the stores this page wrote
import shutil

for path in ['tuning.zarr', 'flat.zarr']:
    shutil.rmtree(path, ignore_errors=True)
```
