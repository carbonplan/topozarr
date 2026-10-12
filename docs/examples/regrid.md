# Regridding

`create_pyramid` only coarsens. It needs a uniform 1-D grid, and it keeps that grid. To change the resolution or extent, or to start from a non-uniform grid, regrid first. This example uses [xarray-regrid](https://github.com/EXCITED-CO2/xarray-regrid) to put one day of `air_temperature` on a 0.5° grid, then builds the pyramid.

```python exec="on" session="regrid"
# hidden: render CarbonPlan-themed light and dark figures inline
import sys

import matplotlib

matplotlib.use("Agg")


sys.path.insert(0, "scripts")
from docs_plot import render  # noqa: E402


# print here is markdown-exec's captured print, not the module's
def show(plot):
    print(render(plot))
```

```python exec="on" source="above" session="regrid" result="python"
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

```python exec="on" source="above" session="regrid" result="python"
from topozarr import create_pyramid

create_pyramid(fine, levels=3, x_dim="lon", y_dim="lat").write("regrid.zarr")
dt = xr.open_datatree("regrid.zarr", engine="zarr", consolidated=False)
print({level: dict(dt[level].sizes) for level in dt.children})
```

```python exec="on" source="above" session="regrid" html="true"
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

```python exec="on" session="regrid"
# hidden: remove the stores this page wrote
import shutil

shutil.rmtree("regrid.zarr", ignore_errors=True)
```
