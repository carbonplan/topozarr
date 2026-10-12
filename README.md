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
# or
pip install topozarr
```

Multiscales are computed by `topozarr-core`, a small Rust kernel installed automatically as a wheel. The `tutorial` extra includes everything needed to run the examples below:

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

Remote stores keep zarr-python's pipeline by default; see [Tips](https://carbonplan.github.io/topozarr/tips/#faster-local-writes) to opt in.

### Example

```python
import xarray as xr
import xproj  # for CRS assignment
from topozarr import create_pyramid

# Load the air_temperature Xarray tutorial dataset
ds = xr.tutorial.open_dataset('air_temperature').drop_encoding()

# Assign a CRS
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
print(ds)
```

```python
pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    method="mean",  # "mean" (default) | "max" | "min" | "sum" | "nearest"
)
print(pyramid.encoding)

# write
pyramid.write("pyramid.zarr")
```

`levels` is the total number of resolution levels, including the original. Level `0` is the original (highest) resolution; by default each subsequent level is coarsened by a factor of 2 per spatial dimension, so the last level is the coarsest. Pass `levels="auto"` to keep adding levels until the smaller spatial dim would drop below `min_dim` (default 256).

`create_pyramid` returns a write plan; `pyramid.write(store)` does the work.

Not every dataset needs overviews! For lower resolution Zarr stores use
`attach_geozarr_metadata` to add GeoZarr attrs and `recommend_encoding`
for chunking/sharding heuristics for web mapping.

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

[Runnable examples](https://carbonplan.github.io/topozarr/examples/quickstart/) cover categorical data, tuning, Icechunk, and Dask. Coming from [ndpyramid](https://github.com/carbonplan/ndpyramid)? See the [comparison](https://carbonplan.github.io/topozarr/tips/#coming-from-ndpyramid).


## Contributing

Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel. See the [contributing docs](https://carbonplan.github.io/topozarr/contributing/) for setup, tests, conformance tests, linting, and demo-data scripts.

## License

MIT — see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org)
