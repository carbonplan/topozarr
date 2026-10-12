# Icechunk and Dask

Write a packed, Dask-backed source to an [Icechunk](https://icechunk.io) repo, then write the same pyramid with Dask. For S3, swap the storage (see [Usage](../usage.md#writing-to-the-cloud)).

## A packed source

`air_temperature` is stored as `int16` with a `scale_factor`; opening it decodes to `float64`, 4x the size. Write it to Zarr packed, with a `_FillValue` and one chunk per year, then open it lazily:

```python exec="on" source="above" session="icechunk" result="text"
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
print(ds.air)
```

`create_pyramid` re-packs decoded input to its stored dtype, so every level stays `int16`:

```python exec="on" source="above" session="icechunk" result="text"
from topozarr import create_pyramid

pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
print(pyramid.level_templates[1].air.dtype, pyramid.level_templates[1].air.attrs["scale_factor"])
```

## Write to Icechunk

`write` takes an Icechunk session store; commit afterwards:

```python exec="on" source="above" session="icechunk" result="text"
import icechunk

repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage("air.icechunk"))
session = repo.writable_session("main")
pyramid.write(session.store, mode="w")
session.commit("write air_temperature pyramid")

dt = xr.open_datatree(repo.readonly_session("main").store, engine="zarr", consolidated=False)
print(dt["1"].air.encoding["dtype"], dt["1"].air.encoding["shards"])
```

## Write with Dask

`Pyramid.write` runs on a local thread pool. For Dask (e.g. a distributed cluster), `as_datatree()` returns a lazy DataTree; pass `pyramid.encoding` to keep the recommended chunks and shards. The shards are sized to divide the Dask chunks, so no rechunk is needed:

```python exec="on" source="above" session="icechunk" result="text"
lazy = pyramid.as_datatree()
lazy.to_zarr("air-dask.zarr", zarr_format=3, consolidated=False, mode="w", encoding=pyramid.encoding)
print(lazy)
```

```python exec="on" session="icechunk"
# hidden: remove the stores this page wrote
import shutil

for path in ['air.zarr', 'air-dask.zarr', 'air.icechunk']:
    shutil.rmtree(path, ignore_errors=True)
```
