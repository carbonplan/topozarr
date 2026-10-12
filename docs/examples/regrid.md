# Regridding

`create_pyramid` only coarsens: it needs a uniform 1-D grid and keeps it. To change the grid (resolution, extent, or a non-uniform source grid), regrid first. This page uses [xarray-regrid](https://github.com/EXCITED-CO2/xarray-regrid) to put one day of `air_temperature` on a 0.5° grid, then builds the pyramid.

```python exec="on" session="regrid"
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

```python exec="on" source="above" session="regrid" result="text"
import xarray as xr
import xarray_regrid  # noqa: F401
import xproj  # noqa: F401
from xarray_regrid import Grid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding().sel(time="2013-01-01")

target = Grid(
    north=75, south=15, west=200, east=330, resolution_lat=0.5, resolution_lon=0.5
).create_regridding_dataset(lat_name="lat", lon_name="lon")
fine = ds.regrid.linear(target).proj.assign_crs(spatial_ref="EPSG:4326")
print(dict(fine.sizes))
```

```python exec="on" source="above" session="regrid" result="text"
from topozarr import create_pyramid

create_pyramid(fine, levels=3, x_dim="lon", y_dim="lat").write("regrid.zarr")
dt = xr.open_datatree("regrid.zarr", engine="zarr", consolidated=False)
print({level: dict(dt[level].sizes) for level in dt.children})
```

```python exec="on" source="above" session="regrid" html="true"
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, level in zip(axes, dt.children):
    dt[level].air.isel(time=0).plot(ax=ax, vmin=230, vmax=305)
show(fig)
```

```python exec="on" session="regrid"
# hidden: remove the stores this page wrote
import shutil

shutil.rmtree("regrid.zarr", ignore_errors=True)
```
