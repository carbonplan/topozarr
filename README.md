<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-dark.svg">
    <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-light.svg" alt="topozarr" width="320">
  </picture>
</p>


### Create Multiscale Zarr stores

Python library companion to the TypeScript web mapping tool [zarr-layer](https://zarr-layer.demo.carbonplan.org/). Use it to create GeoZarr-compliant multiscales / pyramids / overviews for Zarr stores for use with web mapping.

Follows the [zarr-conventions](https://github.com/zarr-conventions):

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names


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

`create_pyramid` is lazy; `write` does the work. `levels` counts all levels: `0` is the original resolution and each level after it is 2x coarser (see [Tuning](https://carbonplan.github.io/topozarr/tuning/) for `levels="auto"`).

Small datasets may not need overviews. Write a single-resolution store with GeoZarr attrs and web-friendly chunks:

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

The [docs](https://carbonplan.github.io/topozarr/usage/) have runnable examples for categorical data, regridding, tuning, Icechunk, and Dask. Coming from [ndpyramid](https://github.com/carbonplan/ndpyramid)? See the [comparison](https://carbonplan.github.io/topozarr/tips/#coming-from-ndpyramid).


## Contributing

Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel. See the [contributing docs](https://carbonplan.github.io/topozarr/contributing/) for setup, tests, conformance tests, linting, and demo-data scripts.

## License

MIT — see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org)
