# Usage

Build a pyramid from the Xarray `air_temperature` tutorial dataset, write it, and open it back.

```python exec="on" session="usage"
# hidden: render matplotlib figures inline
import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")


def show(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    print(f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
```

## Load and assign a CRS

`create_pyramid` needs a CRS; [xproj](https://xproj.readthedocs.io) adds the `.proj` accessor.

```python exec="on" source="above" session="usage" result="text"
import xarray as xr
import xproj  # noqa: F401

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
print(ds)
```

## Plan and write

`create_pyramid` is lazy: it returns a plan and writes nothing until `write`. `levels` is the total number of levels; level `0` is the original resolution and each level after it is coarsened 2x per spatial dimension.

```python exec="on" source="above" session="usage" result="text"
from topozarr import create_pyramid

pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    method="mean",  # "mean" (default) | "max" | "min" | "sum" | "nearest"
)
pyramid.write("pyramid.zarr")
```

For land cover, masks, or other class codes, use `method="nearest"` (see [Categorical data](examples/categorical.md)).

## Open as a DataTree

Each level is a child group with the same extent at half the resolution of the one before:

```python exec="on" source="above" session="usage" result="text"
dt = xr.open_datatree("pyramid.zarr", engine="zarr", consolidated=False)
print(dt)
```

```python exec="on" source="above" session="usage" html="true"
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, level in zip(axes, dt.children):
    dt[level].air.isel(time=0).plot(ax=ax, vmin=230, vmax=305)
show(fig)
```

Open the store in [zarr-layer](https://zarr-layer.demo.carbonplan.org/) to view it on a map.

## Writing to the cloud

`pyramid.write` accepts a local path, an `Obstore` store, or an `Icechunk` store.

=== "Icechunk"

    ```python
    import icechunk

    storage = icechunk.s3_storage(
        bucket="<your_bucket>", prefix="<your_prefix>", from_env=True
    )
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    pyramid.write(session.store, mode="w")
    session.commit("write pyramid")
    ```

=== "Obstore"

    ```python
    from obstore.store import from_url
    from zarr.storage import ObjectStore

    store = ObjectStore(from_url("s3://<your_bucket>/air.zarr", region="us-west-2"))
    pyramid.write(store, mode="w")
    ```

## Single-resolution stores

Small datasets often don't need overviews: `zarr-layer` renders a flat store as long as its chunking is web-friendly. `attach_geozarr_metadata` adds the convention attrs and `recommend_encoding` returns the chunking `create_pyramid` uses per level:

```python exec="on" source="above" session="usage" result="text"
from topozarr import attach_geozarr_metadata, recommend_encoding

flat = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
flat.to_zarr(
    "flat.zarr",
    zarr_format=3,
    consolidated=False,
    encoding=recommend_encoding(flat, x_dim="lon", y_dim="lat"),
)
```

## Visualization hints

Optional. `layer_hints` embeds a default colormap and color range for [zarr-layer](https://zarr-layer.demo.carbonplan.org/); it has no effect on the data.

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

Next: [Tuning](tuning.md) for level counts and chunk sizes, [Examples](examples/categorical.md) for categorical data, regridding, Icechunk, and Dask.

```python exec="on" session="usage"
# hidden: remove the stores this page wrote
import shutil

for path in ["pyramid.zarr", "flat.zarr"]:
    shutil.rmtree(path, ignore_errors=True)
```
