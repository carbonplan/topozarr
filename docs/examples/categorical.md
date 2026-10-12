# Categorical data

Averaging class codes invents classes: the mean of tree cover (`10`) and water (`80`) is `45`. For land cover, masks, and class IDs use `method="nearest"`, which keeps the top-left cell of each window.

This page uses a window of an [ESA WorldCover](https://esa-worldcover.org) 2021 tile over Missoula, Montana, read lazily with [rioxarray](https://corteva.github.io/rioxarray/).

```python exec="on" session="categorical"
# hidden: render matplotlib figures inline
import base64
from io import BytesIO

import os

import matplotlib

matplotlib.use("Agg")
# retry the remote GeoTIFF read on transient network errors
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")


def show(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    print(f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
```

```python exec="on" source="above" session="categorical" result="text"
import rioxarray
import xarray as xr
import xproj  # noqa: F401

url = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "v200/2021/map/ESA_WorldCover_10m_2021_v200_N45W117_Map.tif"
)
landcover = (
    rioxarray.open_rasterio(url, chunks=True)
    .squeeze("band", drop=True)
    .sel(x=slice(-114.2, -114.0), y=slice(46.97, 46.77))
)
# 0 is a WorldCover class, not missing data: drop the GDAL nodata tag
landcover.attrs.pop("_FillValue")
ds = landcover.to_dataset(name="landcover").proj.assign_crs(spatial_ref="EPSG:4326")
print(ds)
```

Build a pyramid with each method and open the coarsest level (8x). `mask_and_scale=False` keeps class codes as `uint8`:

```python exec="on" source="above" session="categorical" result="text"
from topozarr import create_pyramid

coarse = {}
for method in ["nearest", "mean"]:
    create_pyramid(ds, levels=4, method=method).write(f"{method}.zarr")
    dt = xr.open_datatree(f"{method}.zarr", engine="zarr", consolidated=False, mask_and_scale=False)
    coarse[method] = dt["3"].landcover

coarse = xr.concat(coarse.values(), dim=xr.Variable("method", list(coarse)))
print(coarse)
```

`nearest` only returns real classes. `mean` keeps the dtype but fills class boundaries with codes that are not classes:

```python exec="on" source="above" session="categorical" result="text"
classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
print((~coarse.isin(classes)).mean(["x", "y"]).to_series())
```

```python exec="on" source="above" session="categorical" html="true"
from matplotlib.colors import BoundaryNorm, ListedColormap

colors = ["#000000", "#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000",
          "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0"]
norm = BoundaryNorm([c - 0.5 for c in classes] + [100.5], len(colors))

grid = coarse.plot.imshow(col="method", cmap=ListedColormap(colors), norm=norm, size=5, add_colorbar=False)
show(grid.fig)
```

The trade-off: `nearest` samples one cell per window, so a rare class that never lands on a window corner can shrink or vanish at coarse zoom.

```python exec="on" session="categorical"
# hidden: remove the stores this page wrote
import shutil

for path in ['nearest.zarr', 'mean.zarr']:
    shutil.rmtree(path, ignore_errors=True)
```
