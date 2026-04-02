# topozarr - lightweight multiscale zarr pyramids

Python library to create multiscale Zarr pyramids for usage with [zarr-layer](https://zarr-layer.demo.carbonplan.org/).

Attempts to follow the [GeoZarr spec](https://github.com/zarr-developers/geozarr-spec).

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-experimental/geo-proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

**Warning: experimental**

## Usage

#### Installation

You can install the tutorial optional dependency group to run this example.

```bash
uv add 'topozarr[tutorial]'
# or
pip install 'topozarr[tutorial]'
```

#### Example
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

#### Chunking

A recommended encoding is returned with the pyramid. You can access it with `.pyramid.encoding`. There are some basic heuristics to try to get chunk sizes of ~500KB for web visualization and shard sizes 4 times the size (configurable). You can tune the size of the shards with the `chunks_per_shard` parameter (default: `4`, giving `16` chunks per shard and ~8MB shards). Valid values are powers of 2: `1, 2, 4, 8, 16, 32`. Larger shards increase memory usage, but decrease the task graph overhead if using Dask.


| `chunks_per_shard` | chunks/shard | approx shard size |
|--------------------|--------------|-------------------|
| 1 | 1 | ~500KB |
| 4 | 16 | ~8MB (default) |
| 8 | 64 | ~32MB |
| 16 | 256 | ~128MB |


```python
pyramid = create_pyramid(ds, levels=8, x_dim="lon", y_dim="lat")
```

Pass `chunks_per_shard=None` to disable sharding entirely.

```python
# Optional: Write to Zarr
# !pip install obstore zarr
from obstore.store import from_url
from zarr.storage import ObjectStore


store = from_url(url = "<add_your_bucket_url>", region="<add_your_region>")
zstore = ObjectStore(store) 
pyramid.dt.to_zarr(zstore, mode="w", encoding = pyramid.encoding, zarr_format=3)
```

```python
# Optional: Write to Icechunk
# !pip install icechunk 
import icechunk

storage = icechunk.s3_storage(bucket="<add_your_bucket_name>", prefix="<add_your_prefix>", from_env=True)
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")

store = from_url(url = "<add_your_bucket_url>", region="<add_your_region>")
pyramid.dt.to_zarr(session.store, mode="w", encoding = pyramid.encoding, consolidated=False)
```



## Development

This project uses `uv` for dependency management, `pytest` and `hypothesis` for testing and `ruff` for linting. 


### Sync development environment 

```python
uv sync --all-extras
```


### Run linter

```python
uv run pre-commit run all-files
```

### Run tests
```
uv run pytest tests/
```

### Run conformance tests - test against geozarr spec using geozarr-toolkit (requires `geozarr-toolkit`)
```
uv sync --group conformance
uv run pytest tests/ -m conformance
```

## License

> [!IMPORTANT]
> This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About Us

CarbonPlan is a nonprofit organization that uses data and science for climate action. We aim to improve the transparency and scientific integrity of climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/{repo-name}/issues/new) or [sending us an email](mailto:hello@carbonplan.org)

