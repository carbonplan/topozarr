# API Reference

The public API:

- `create_pyramid`: builds a write plan.
- `Pyramid`: holds the plan; `write` or `as_datatree` materializes it.
- `attach_geozarr_metadata`: adds geozarr convention attrs without building a pyramid.
- `recommend_encoding`: returns the chunk/shard encoding for a flat dataset.
- `ZarrLayerVarConfig`: optional visualization hints.

`CoarseningMethod` is the `Literal["mean", "max", "min", "sum", "nearest"]` alias accepted by `create_pyramid(method=...)`. `nearest` decimates (corner-pick) for categorical data. A test keeps it equal to `topozarr_core.METHODS`, the installed kernel's own list.

::: topozarr.coarsen.create_pyramid

::: topozarr.pyramid.Pyramid

::: topozarr.geozarr.attach_geozarr_metadata

::: topozarr.metadata.recommend_encoding

::: topozarr.metadata.ZarrLayerVarConfig
