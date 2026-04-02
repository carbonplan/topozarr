# topozarr - lightweight multiscale zarr pyramids

Python library to create multiscale Zarr pyramids for usage with [zarr-layer](https://zarr-layer.demo.carbonplan.org/).

Attempts to follow the [GeoZarr spec](https://github.com/zarr-developers/geozarr-spec).

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-experimental/geo-proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

**Warning: experimental**

## Usage

### Installation

You can install the tutorial optional dependency group to run this example.

```bash
uv add 'topozarr[tutorial]'
# or
pip install 'topozarr[tutorial]'
```

### Example

```python
import xarray as xr
import xproj # for crs assignment
from topozarr.coarsen import create_pyramid

# Load the air_temperature Xarray tutorial dataset
ds = xr.tutorial.open_dataset('air_temperature', chunks="auto")
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
print(pyramid.dt)
```

### Visualization hints

Use `layer_hints` to embed colormap and color range hints for [zarr-layer](https://zarr-layer.demo.carbonplan.org/) directly in the pyramid metadata:

```python
from topozarr.metadata import ZarrLayerVarConfig

pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    layer_hints={"air": ZarrLayerVarConfig(colormap="blues", clim=[230, 310])},
)
```

These are written into the root `zarr-layer` metadata key and are optional — omitting `layer_hints` has no effect on the pyramid structure or encoding.

### Chunking

`create_pyramid` returns a `Pyramid` with two attributes: `pyramid.dt` (the `DataTree`) and `pyramid.encoding` (recommended chunk and shard sizes per variable per level). Always pass `pyramid.encoding` as the `encoding` argument when writing — this is what applies the chunking strategy to the output store.

```python
# Inspect the recommended encoding before writing
print(pyramid.encoding)
```

The heuristics target ~500KB chunks for web visualization. You can tune shard size with `chunks_per_shard` (default: `4`, giving 16 chunks per shard and ~8MB shards). Valid values are powers of 2: `1, 2, 4, 8, 16, 32`. Larger shards reduce task graph overhead when using Dask but increase memory usage.

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|--------------|-------------------|
| 1 | 1 | ~500KB |
| 4 | 16 | ~8MB (default) |
| 8 | 64 | ~32MB |
| 16 | 256 | ~128MB |

Pass `chunks_per_shard=None` to disable sharding entirely.

### Writing

Always pass `pyramid.encoding` to apply the recommended chunking:

```python
# Write to Zarr
from obstore.store import from_url
from zarr.storage import ObjectStore

store = ObjectStore(from_url(url="<your_bucket_url>", region="<your_region>"))
pyramid.dt.to_zarr(store, mode="w", encoding=pyramid.encoding, zarr_format=3)
```

```python
# Write to Icechunk
import icechunk

storage = icechunk.s3_storage(bucket="<your_bucket>", prefix="<your_prefix>", from_env=True)
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")
pyramid.dt.to_zarr(session.store, mode="w", encoding=pyramid.encoding, consolidated=False)
```

## Contributing

Clone the repo and install with the `test` dependency group:

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

To regenerate the demo datasets in S3 (requires AWS credentials), install the `demo` extra and run the build script:

```bash
uv sync --extra demo
uv run python scripts/build_demo_data.py --help
```

## License

> [!IMPORTANT]
> This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/topozarr/issues/new) or [sending us an email](mailto:hello@carbonplan.org)
