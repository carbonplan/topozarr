# topozarr

Create multiscale Zarr stores for web visualization.

Built for use with [zarr-layer](https://zarr-layer.demo.carbonplan.org/). Tries to follow the GeoZarr spec, which is composed of these Zarr conventions:

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/geo-proj) — coordinate reference system (CRS)
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
