# API Reference

The public API is four objects: `create_pyramid` builds a write plan, `Pyramid` holds it, `attach_geozarr_metadata` adds geozarr convention attrs without building a pyramid, and `ZarrLayerVarConfig` carries optional visualization hints. `CoarseningMethod` is the `Literal["mean", "max", "min", "sum"]` alias accepted by `create_pyramid(method=...)`.

::: topozarr.coarsen.create_pyramid

::: topozarr.pyramid.Pyramid

::: topozarr.geozarr.attach_geozarr_metadata

::: topozarr.metadata.ZarrLayerVarConfig

## Engine

Lower-level streaming drivers used by `Pyramid.write`; useful when writing custom pipelines on top of the kernel.

::: topozarr.engine.downsample_level

::: topozarr.engine.copy_array

::: topozarr.engine.default_max_workers
