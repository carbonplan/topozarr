# Usage

This page walks through the full workflow: build a pyramid from the Xarray `air_temperature` tutorial dataset, write it to a Zarr store, and open it back up.

```python exec="on" session="usage"
# hidden: render CarbonPlan-themed light and dark figures inline
import sys

import matplotlib

matplotlib.use("Agg")


sys.path.insert(0, "scripts")
from docs_plot import html_repr, render  # noqa: E402


# print here is markdown-exec's captured print, not the module's
def show(plot):
    print(render(plot))
```

## Load and assign a CRS

`create_pyramid` needs to know the dataset's coordinate reference system (CRS). [xproj](https://xproj.readthedocs.io) adds a `.proj` accessor for assigning one.

```python exec="on" source="above" session="usage" html="true"
import xarray as xr
import xproj  # noqa: F401

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
ds
print(html_repr(ds))  # markdown-exec: hide
```

## Plan and write

`create_pyramid` is lazy. It returns a plan and writes nothing until you call `write`. `levels` is the total number of levels. Level `0` is the original resolution, and each level after it is 2x coarser along each spatial dimension.

<div class="tz-figure">
--8<-- "docs/assets/figures/levels.svg"
</div>

```python exec="on" source="above" session="usage"
from topozarr import create_pyramid

pyramid = create_pyramid(
    ds,
    levels=3,
    x_dim="lon",
    y_dim="lat",
    method="mean",  # "mean" (default) | "max" | "min" | "sum" | "nearest"
)
pyramid.write("pyramid.zarr")
```

For land cover, masks, or other class codes, use `method="nearest"` so that averaging doesn't invent new classes (see [Categorical data](examples/categorical.md)).

## Open as a DataTree

Each level is a child group that covers the same extent at half the resolution of the level before it:

```python exec="on" source="above" session="usage" html="true"
dt = xr.open_datatree("pyramid.zarr", engine="zarr", consolidated=False)
dt
print(html_repr(dt))  # markdown-exec: hide
```

```python exec="on" source="above" session="usage" html="true"
import matplotlib.pyplot as plt


def plot(mode):
    levels = ["0", str(len(dt.children) - 1)]  # finest and coarsest
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, level in zip(axes, levels):
        im = dt[level].air.isel(time=0).plot(
            ax=ax, vmin=230, vmax=305, cmap=f"warm_{mode}", add_colorbar=False
        )
        ax.set_title(f"level {level}")
    fig.colorbar(im, ax=axes, label="air temperature [K]")
    return fig


show(plot)
```

Open the store in [zarr-layer](https://zarr-layer.demo.carbonplan.org/) to view it on a map.

## Choosing a store

`pyramid.write` accepts a local path, an Obstore store, or an Icechunk store. Obstore and Icechunk can write to either local disk or cloud object storage.

=== "Local path"

    ```python
    pyramid.write("pyramid.zarr")
    ```

=== "Icechunk"

    ```python
    import icechunk

    storage = icechunk.s3_storage(
        bucket="<your_bucket>", prefix="<your_prefix>", from_env=True
    )
    # local: storage = icechunk.local_filesystem_storage("pyramid.icechunk")
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
    # local: store = ObjectStore(from_url("file:///absolute/path/to/air.zarr", mkdir=True))
    pyramid.write(store, mode="w")
    ```

## Single-resolution stores

Small datasets often don't need a pyramid. `zarr-layer` can render a single-resolution store as long as its chunks are sized for the web. `attach_geozarr_metadata` adds the GeoZarr attributes, and `recommend_encoding` returns the same chunking that `create_pyramid` uses for each level:

```python exec="on" source="above" session="usage"
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

`layer_hints` is optional. It embeds a default colormap and color range for [zarr-layer](https://zarr-layer.demo.carbonplan.org/) and has no effect on the data.

```python
from topozarr.metadata import ZarrLayerVarConfig

pyramid = create_pyramid(
    ds,
    levels=2,
    x_dim="lon",
    y_dim="lat",
    layer_hints={"air": ZarrLayerVarConfig(colormap="warm", clim=[230, 310])},
)
```

Next, see [Tuning](tuning.md) for level counts and chunk sizes, or the [examples](examples/categorical.md) for categorical data, regridding, Icechunk, and Dask.

```python exec="on" session="usage"
# hidden: remove the stores this page wrote
import shutil

for path in ["pyramid.zarr", "flat.zarr"]:
    shutil.rmtree(path, ignore_errors=True)
```
