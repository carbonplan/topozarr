<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-dark.svg">
    <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-light.svg" alt="topozarr" width="320">
  </picture>
</p>


### Create multiscale Zarr stores for web maps

`topozarr` is a Python library that turns an Xarray dataset into a multiscale Zarr store: a pyramid of progressively coarser copies of the data, so a web map can load only the resolution it needs. It is the companion to [`zarr-layer`](https://zarr-layer.demo.carbonplan.org/), our TypeScript library for rendering Zarr on MapLibre and Mapbox maps.

Stores follow three [Zarr conventions](https://github.com/zarr-conventions) that make up GeoZarr:

- [multiscales](https://github.com/zarr-conventions/multiscales): pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/proj): coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial): affine transform, bounding box, and dimension names


### Installation

```bash
uv add topozarr

# optional extras
uv add 'topozarr[tutorial]'  # deps for the examples
uv add 'topozarr[zarrs]'     # faster local writes (Rust codec pipeline)
```

```bash
pip install topozarr

# optional extras
pip install 'topozarr[tutorial]'  # deps for the examples
pip install 'topozarr[zarrs]'     # faster local writes (Rust codec pipeline)
```

### Example

```python
import xarray as xr
import xproj  # for CRS assignment
from topozarr import create_pyramid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    method="mean",  # "mean" (default) | "max" | "min" | "sum" | "nearest"
)
pyramid.write("pyramid.zarr")
```

`create_pyramid` is lazy: it builds a plan, and `write` does the work. `levels` is the total number of levels. Level `0` is the original resolution, and each level after it is 2x coarser. To let `topozarr` pick the count, see `levels="auto"` in [Tuning](https://carbonplan.github.io/topozarr/tuning/).

Small datasets may not need a pyramid at all. In that case, write a single-resolution store with GeoZarr attributes and web-friendly chunks:

```python
from topozarr import attach_geozarr_metadata, recommend_encoding

ds = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
ds.to_zarr(
    "flat.zarr",
    zarr_format=3,
    consolidated=False,
    encoding=recommend_encoding(ds, x_dim="lon", y_dim="lat"),
)
```

### More

The [documentation](https://carbonplan.github.io/topozarr/usage/) has runnable examples for categorical data, regridding, tuning, Icechunk, and Dask. If you're coming from [`ndpyramid`](https://github.com/carbonplan/ndpyramid), see the [comparison](https://carbonplan.github.io/topozarr/tips/#coming-from-ndpyramid).


## Contributing

Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel. See the [contributing guide](https://carbonplan.github.io/topozarr/contributing/) for setup, tests, linting, and demo data scripts.

## License

MIT. See the [LICENSE](LICENSE) file for details.

## About us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
