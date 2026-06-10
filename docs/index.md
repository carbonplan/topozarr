# topozarr

Lightweight multiscale Zarr pyramids for web visualization.

`topozarr` creates [multiscale Zarr](https://github.com/zarr-conventions/multiscales) pyramids from georeferenced `xarray.Dataset` objects. Pyramids are written with `zarr-python` — no Dask required. The coarsening kernel is `topozarr-core`, a small Rust extension installed automatically as a wheel.

Built for use with [zarr-layer](https://zarr-layer.demo.carbonplan.org/). Follows these Zarr conventions:

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/geo-proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

> **Warning: experimental** — APIs may change without notice.

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
