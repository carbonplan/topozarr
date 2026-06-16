<p align="center">
  <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/logo_512x512.png" alt="topozarr" width="200">
</p>


Python library to create multiscale Zarr stores for usage with [zarr-layer](https://zarr-layer.demo.carbonplan.org/).

Tries to match the GeoZarr spec, which is composed of these [zarr-conventions](https://github.com/zarr-conventions):

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/geo-proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

**Warning: experimental**

## Usage

### Installation

```bash
uv add topozarr
# or
pip install topozarr
```

Multiscales are computed by `topozarr-core`, a small Rust kernel (installed automatically as a wheel), and written with `zarr-python`. For Dask-distributed workflows, `pyramid.as_datatree()` returns a lazy `xr.DataTree` you can write yourself. The `tutorial` extra should include everything needed to run the examples below:

```bash
uv add 'topozarr[tutorial]'
# or
pip install 'topozarr[tutorial]'
```

### Example

```python
import xarray as xr
import xproj # for crs assignment
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
    method="mean",  # "mean" (default) | "max" | "min" | "sum"
)
print(pyramid.encoding)

# write
pyramid.write("pyramid.zarr")
```

`levels` is the total number of resolution levels, including the original. Level `0` is the original (highest) resolution; by default each subsequent level is coarsened by a factor of 2 per spatial dimension, so the last level is the coarsest.

`create_pyramid` returns a write plan; `pyramid.write(store)` does the work.

### Documentation

Full docs at **[carbonplan.github.io/topozarr](https://carbonplan.github.io/topozarr/)**:

- [Usage](https://carbonplan.github.io/topozarr/usage/) — sparse pyramids (`factors`), visualization hints (`layer_hints`), chunking and sharding (and why sharding rules), object-storage / Icechunk backends, and the experimental `io="rust"` write path.
- [Design](https://carbonplan.github.io/topozarr/design/) — the plan/execute split, chunk and shard heuristics, streaming memory model, and Rust kernel semantics.

## Contributing

Clone the repo and install with the `test` dependency group.

```bash
git clone https://github.com/carbonplan/topozarr
cd topozarr
uv sync --group test
```

Run tests:

```bash
uv run pytest -n auto
```

Run conformance tests against the GeoZarr spec (requires the `conformance` group):

```bash
uv sync --group conformance
uv run pytest -n auto -m conformance
```

Lint and format:

```bash
uv run pre-commit run --all-files
```

To regenerate the demo datasets in S3 (requires AWS credentials), install the `tutorial` extra and run the build script:

```bash
uv sync --extra tutorial
uv run python scripts/build_demo_data.py --help
```

## License

> [!IMPORTANT]
> This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org)
