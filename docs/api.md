# API reference

The public API has five parts:

- `create_pyramid` builds a write plan.
- `Pyramid` holds the plan. `write` or `as_datatree` turns it into data.
- `attach_geozarr_metadata` adds GeoZarr attributes without building a pyramid.
- `recommend_encoding` returns the chunk and shard encoding for a single-resolution dataset.
- `ZarrLayerVarConfig` holds optional visualization hints.

`CoarseningMethod` is the `Literal["mean", "max", "min", "sum", "nearest"]` alias accepted by `create_pyramid(method=...)`. `nearest` keeps the top-left cell of each window, for categorical data. A test keeps the alias equal to `topozarr_core.METHODS`, the installed kernel's own list.

::: topozarr.coarsen.create_pyramid

::: topozarr.pyramid.Pyramid

::: topozarr.geozarr.attach_geozarr_metadata

::: topozarr.metadata.recommend_encoding

::: topozarr.metadata.ZarrLayerVarConfig
