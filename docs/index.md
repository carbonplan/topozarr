---
hide:
  - toc
---

# topozarr

Create multiscale Zarr stores for web visualization.

Built for use with [zarr-layer](https://zarr-layer.demo.carbonplan.org/). Follows the [zarr-conventions](https://github.com/zarr-conventions):

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

!!! warning "Experimental"
    APIs may change without notice.

## Installation

```bash
uv add topozarr
# or
pip install topozarr
```

The `tutorial` extra includes everything needed to run the examples in the [Usage](usage.md) guide:

```bash
uv add 'topozarr[tutorial]'
# or
pip install 'topozarr[tutorial]'
```

For faster local writes, add the `zarrs` extra. `Pyramid.write` then uses the Rust [zarrs](https://github.com/zarrs/zarrs-python) codec pipeline automatically for local stores:

```bash
uv add 'topozarr[zarrs]'
# or
pip install 'topozarr[zarrs]'
```

Remote stores keep zarr-python's pipeline until [zarrs-python#139](https://github.com/zarrs/zarrs-python/issues/139) lands. To opt in anyway, set it yourself; `write` leaves an explicitly configured pipeline alone:

```python
import zarr
import zarrs

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```
