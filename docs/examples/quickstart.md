# Quickstart

Build a pyramid from the Xarray `air_temperature` tutorial dataset, write it, and open it back as a DataTree. Every code block on this page runs when the docs are built.

```python exec="on" session="quickstart"
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

```python exec="on" source="above" session="quickstart" result="text"
import xarray as xr
import xproj  # noqa: F401

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")
print(ds)
```

## Plan and write

`create_pyramid` is lazy: it returns a plan with per-level chunk/shard `encoding` and writes nothing until `write`.

```python exec="on" source="above" session="quickstart" result="text"
from topozarr import create_pyramid

pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
print(pyramid.encoding)

pyramid.write("quickstart.zarr")
```

## Open as a DataTree

Each level is a child group with the same extent at half the resolution of the one before:

```python exec="on" source="above" session="quickstart" result="text"
dt = xr.open_datatree("quickstart.zarr", engine="zarr", consolidated=False)
print(dt)
```

```python exec="on" source="above" session="quickstart" html="true"
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, level in zip(axes, dt.children):
    dt[level].air.isel(time=0).plot(ax=ax, vmin=230, vmax=305)
show(fig)
```

Open the store in [zarr-layer](https://zarr-layer.demo.carbonplan.org/) to view it on a map.

```python exec="on" session="quickstart"
# hidden: remove the stores this page wrote
import shutil

for path in ['quickstart.zarr']:
    shutil.rmtree(path, ignore_errors=True)
```
