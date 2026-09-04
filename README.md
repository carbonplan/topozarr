<p align="center">
  <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/topozarr_logo_name.png" alt="topozarr" width="200">
</p>


### Create Multiscale Zarr stores

Python library companion to the TypeScript web mapping tool [zarr-layer](https://zarr-layer.demo.carbonplan.org/). Use it to create GeoZarr-compliant multiscales / pyramids / overviews for Zarr stores for use with web mapping.

Follows the [zarr-conventions](https://github.com/zarr-conventions):

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/geo-proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

**Warning: experimental**


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

`levels` is the total number of resolution levels, including the original. Level `0` is the original (highest) resolution; by default each subsequent level is coarsened by a factor of 2 per spatial dimension, so the last level is the coarsest.

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

### Coming from ndpyramid
The library [ndpyramid](https://github.com/carbonplan/ndpyramid) also builds multiscale Zarr stores. However, it was built as a companion for [carbonplan-maps](https://github.com/carbonplan/maps), which requires the source data to be reprojected to EPSG:3857, square (e.g. 128x128), slippy-map-tile-compliant shapes. The newer mapping library, [zarr-layer](https://zarr-layer.demo.carbonplan.org/), relaxes these requirements significantly, which simplifies multiscales creation and allows topozarr to be much simpler and more flexible. This project is essentially a coarsen call and some metadata in a trenchcoat.


| ndpyramid | topozarr |
| --- | --- |
| `pyramid_coarsen` | `create_pyramid(ds, levels=...)` |
| `pyramid_reproject` | no equivalent — reproject upstream, then `create_pyramid` |
| `pyramid_regrid` | no equivalent — regrid upstream, then `create_pyramid` |

If you need a different grid, do it before and hand
the result to `create_pyramid`.


## Contributing

Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel. See the [contributing docs](https://carbonplan.github.io/topozarr/contributing/) for setup, tests, conformance tests, linting, and demo-data scripts.

## License

MIT — see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org)
