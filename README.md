<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-dark.svg">
    <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/assets/topozarr-lockup-on-light.svg" alt="topozarr" width="320">
  </picture>
</p>


<p align="center">
  <b>Create multiscale Zarr stores for web visualization.</b>
  <br>
  <a href="https://carbonplan.github.io/topozarr/">Documentation</a> ·
  <a href="https://carbonplan.github.io/topozarr/examples/categorical/">Examples</a> ·
  <a href="https://carbonplan.github.io/topozarr/releases/">Release notes</a>
</p>

`topozarr` turns an Xarray dataset into a multiscale [Zarr v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) store, with each level 2x coarser than the last.

The stores follow the [GeoZarr](https://github.com/zarr-developers/geozarr-spec) conventions and are built for [`zarr-layer`](https://zarr-layer.demo.carbonplan.org/), which renders them directly in the browser.

## Installation

```bash
uv add topozarr
```

Or with pip: `pip install topozarr`.

## Quick example

```python
import xarray as xr
import xproj  # for CRS assignment
from topozarr import create_pyramid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
pyramid.write("pyramid.zarr")
```

See the [usage guide](https://carbonplan.github.io/topozarr/usage/) for cloud stores, tuning, and more.

## License

MIT. See the [LICENSE](LICENSE) file for details.

## About us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
