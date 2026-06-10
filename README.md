<p align="center">
  <img src="https://raw.githubusercontent.com/carbonplan/topozarr/main/docs/logo_512x512.png" alt="topozarr" width="200">
</p>

# lightweight multiscale zarr pyramids

Python library to create multiscale Zarr pyramids for usage with [zarr-layer](https://zarr-layer.demo.carbonplan.org/).

Follows these [zarr-conventions](https://github.com/zarr-conventions):

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

Pyramids are computed by `topozarr-core`, a small Rust kernel (installed automatically as a wheel), and written with `zarr-python` — no Dask involved. The `tutorial` extra includes everything needed to run the examples below:

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

# compute and write all levels
pyramid.write("pyramid.zarr")
```

`levels` is the total number of resolution levels, including the original. Level `0` is the original (highest) resolution; each subsequent level is coarsened by a factor of 2 per spatial dimension, so the last level is the coarsest.

`create_pyramid` returns a write plan; `pyramid.write(store)` does the work: level 0 is copied from the source dataset, then each level is block-reduced from the previously written one, streaming shard-sized regions through the Rust kernel on a thread pool (tune with `max_workers`). Reduction semantics match `xarray.coarsen(boundary="trim")` exactly, including skipna handling of NaN / `_FillValue` nodata.

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

`pyramid.encoding` holds the chunk and shard sizes per variable per level; `pyramid.write` applies them automatically.

```python
# Inspect the encoding before writing
print(pyramid.encoding)
```

The heuristics target ~500KB chunks for web visualization. You can tune shard size with `chunks_per_shard`, the number of chunks per shard *along each spatial dimension* (default: `4`, giving 4×4 = 16 chunks per shard and ~8MB shards). Valid values are powers of 2: `1, 2, 4, 8, 16, 32`. Shards are also the unit of work during pyramid generation, so larger shards mean fewer, bigger reads/writes and higher memory usage.

| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|--------------|-------------------|
| 1 | 1 | ~500KB |
| 4 | 16 | ~8MB (default) |
| 8 | 64 | ~32MB |
| 16 | 256 | ~128MB |

Pass `chunks_per_shard=None` to disable sharding entirely.

### Writing

`pyramid.write` accepts anything zarr-python can open — a local path, an `ObjectStore`, or an icechunk session store:

```python
# Write to object storage
from obstore.store import from_url
from zarr.storage import ObjectStore

store = ObjectStore(from_url(url="s3://carbonplan-scratch/topozarr/aira.zarr", region="us-west-2"))
pyramid.write(store, mode="w")
```

```python
# Write to Icechunk
import icechunk

storage = icechunk.s3_storage(bucket="<your_bucket>", prefix="<your_prefix>", from_env=True)
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")
pyramid.write(session.store, mode="w")
session.commit("write pyramid")
```

### Optional: zarrs codec pipeline

Compression codec work can optionally be routed through the Rust [zarrs](https://github.com/zarrs/zarrs-python) codec pipeline. It plugs in at the codec layer, so it works with any store backend (filesystem, `ObjectStore`, icechunk):

```bash
uv add zarrs
```

```python
import zarr
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

## Contributing

Clone the repo and install with the `test` dependency group. Building from source requires a [Rust toolchain](https://rustup.rs) for the `topozarr-core` kernel (in `core/`):

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
