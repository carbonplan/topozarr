# API Reference

The public API is five objects: `create_pyramid` builds a write plan, `Pyramid` holds it, `attach_geozarr_metadata` adds geozarr convention attrs without building a pyramid, `recommend_encoding` returns the chunk/shard encoding for a flat dataset, and `ZarrLayerVarConfig` carries optional visualization hints. `CoarseningMethod` is the `Literal["mean", "max", "min", "sum", "nearest"]` alias accepted by `create_pyramid(method=...)`, kept equal to `topozarr_core.METHODS` (the installed kernel's own list, which validation checks against) by a test; `nearest` decimates (corner-pick) for categorical data.

::: topozarr.coarsen.create_pyramid

::: topozarr.pyramid.Pyramid

::: topozarr.geozarr.attach_geozarr_metadata

::: topozarr.metadata.recommend_encoding

::: topozarr.metadata.ZarrLayerVarConfig
