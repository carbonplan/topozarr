# Icechunk and Dask

This example starts from a packed, Dask-backed source and writes a pyramid two ways: to an [Icechunk](https://icechunk.io) repository with `write`, and with Dask. It writes to local disk. To write to S3 instead, swap the storage (see [Usage](../usage.md#choosing-a-store)).

## A packed source

```python exec="on" session="icechunk"
# hidden: xarray's HTML repr for displayed objects
import sys

sys.path.insert(0, "scripts")
from docs_plot import html_repr  # noqa: E402
```

`air_temperature` is stored as `int16` with a `scale_factor`. Opening it decodes the values to `float64`, which is four times the size. Write it to Zarr in its packed form, with a `_FillValue` and one chunk per year, then open it lazily:

```python exec="on" source="above" session="icechunk" html="true"
import xarray as xr
import xproj  # noqa: F401

xr.tutorial.open_dataset("air_temperature").chunk(time=730).to_zarr(
    "air.zarr",
    zarr_format=3,
    consolidated=False,
    mode="w",
    encoding={"air": {"dtype": "int16", "scale_factor": 0.01, "_FillValue": -9999}},
)

ds = xr.open_zarr("air.zarr", consolidated=False).proj.assign_crs(spatial_ref="EPSG:4326")
ds.air
print(html_repr(ds.air))  # markdown-exec: hide
```

`create_pyramid` re-packs decoded input to its stored dtype, so every level stays `int16`:

```python exec="on" source="above" session="icechunk" result="python"
from topozarr import create_pyramid

pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
print(pyramid.level_templates[1].air.dtype, pyramid.level_templates[1].air.attrs["scale_factor"])
```

## Write to Icechunk

`write` accepts an Icechunk session store. Commit the session afterward:

```python exec="on" source="above" session="icechunk" result="python"
import icechunk

repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage("air.icechunk"))
session = repo.writable_session("main")
pyramid.write(session.store, mode="w")
session.commit("write air_temperature pyramid")

dt = xr.open_datatree(repo.readonly_session("main").store, engine="zarr", consolidated=False)
print(dt["1"].air.encoding["dtype"], dt["1"].air.encoding["shards"])
```

## Write with Dask

`Pyramid.write` runs on a local thread pool. To use Dask instead (for example, on a distributed cluster), call `as_datatree()` to get a lazy DataTree, and pass `pyramid.encoding` to `to_zarr` to keep the recommended chunks and shards. The shards are sized to divide evenly into the Dask chunks, so no rechunking is needed:

```python exec="on" source="above" session="icechunk" html="true"
lazy = pyramid.as_datatree()
lazy.to_zarr("air-dask.zarr", zarr_format=3, consolidated=False, mode="w", encoding=pyramid.encoding)
lazy
print(html_repr(lazy))  # markdown-exec: hide
```

```python exec="on" session="icechunk"
# hidden: remove the stores this page wrote
import shutil

for path in ['air.zarr', 'air-dask.zarr', 'air.icechunk']:
    shutil.rmtree(path, ignore_errors=True)
```
