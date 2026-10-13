# Dask

`Pyramid.write` runs on a local thread pool. To use Dask instead, for example on a distributed cluster, call `as_datatree()` to get a lazy DataTree and write it with `to_zarr`.

## A lazy source

```python exec="on" session="dask"
# hidden: xarray's HTML repr for displayed objects
import sys

sys.path.insert(0, "scripts")
from docs_plot import html_repr  # noqa: E402
```

Open `air_temperature` with one Dask chunk per year:

```python exec="on" source="above" session="dask" html="true"
import xarray as xr
import xproj  # noqa: F401

ds = xr.tutorial.open_dataset("air_temperature", chunks={"time": 730}).drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
ds.air
print(html_repr(ds.air))  # markdown-exec: hide
```

## Write with Dask

Plan the pyramid as usual. `as_datatree()` returns every level as a lazy DataTree, and `pyramid.encoding` holds the recommended chunks and shards. The shards are sized to divide evenly into the Dask chunks, so no rechunking is needed:

```python exec="on" source="above" session="dask" html="true"
from topozarr import create_pyramid

pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
lazy = pyramid.as_datatree()
lazy.to_zarr("air-dask.zarr", zarr_format=3, consolidated=False, mode="w", encoding=pyramid.encoding)
lazy
print(html_repr(lazy))  # markdown-exec: hide
```

To run on a cluster, start a `dask.distributed.Client` before calling `to_zarr`.

```python exec="on" session="dask"
# hidden: remove the store this page wrote
import shutil

shutil.rmtree("air-dask.zarr", ignore_errors=True)
```
