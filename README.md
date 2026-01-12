# topozarr - lightweight multiscale zarr pyramids


**Warning: experimental**

## Usage

#### Installation

```bash
uv add topozarr
# or
pip install topozarr
```

#### Example
```python
# !pip install netcdf4 pooch 
import xarray as xr
import xproj # for crs assignment
from topozarr.coarsen import create_pyramid

# Load the air_temp tutorial xarray dataset
ds = xr.tutorial.open_dataset('air_temperature', chunks={})
ds = ds.proj.assign_crs(spatial_ref_crs={"EPSG":4326})
print(ds)
```

```python
pyramid = create_pyramid(
    ds, 
    levels=2, 
    x_dim="lon", 
    y_dim="lat", 
    spec="ndpyramid" # or "zarr-multiscales"
)
print(pyramid.encoding)
print(pyramid.dt)
```

```python
# Optional: Write to Zarr
!pip install obstore zarr
from obstore.store import from_url
from zarr.storage import ObjectStore


store = from_url(url = "<add_your_bucket_url>", region="<add_your_region>")
zstore = ObjectStore(store) 
result_GEOG.dt.to_zarr(zstore, mode="w", encoding = pyramid.encoding, zarr_format=3)
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
