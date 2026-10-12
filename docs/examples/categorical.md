# Categorical data

Land cover maps, masks, and other categorical data store class codes, not quantities, so averaging them invents classes that don't exist. For example, a window with one tree cover cell (`10`) and three water cells (`80`) averages to `62`, which isn't a class at all. For this kind of data, use `method="nearest"`, which keeps the top-left cell of each window.

<div class="tz-figure">
--8<-- "docs/assets/figures/categorical.svg"
</div>

This example uses a small window of an [ESA WorldCover](https://esa-worldcover.org) 2021 tile over Missoula, Montana, read lazily with [rioxarray](https://corteva.github.io/rioxarray/). We build a pyramid with both `nearest` and `mean` and compare the results.

```python exec="on" session="categorical"
# hidden: render CarbonPlan-themed light and dark figures inline
import os
import sys

import matplotlib

matplotlib.use("Agg")
# retry the remote GeoTIFF read on transient network errors
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")


sys.path.insert(0, "scripts")
from docs_plot import html_repr, render  # noqa: E402


# print here is markdown-exec's captured print, not the module's
def show(plot):
    print(render(plot))
```

```python exec="on" source="above" session="categorical" html="true"
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
    .sel(x=slice(-114.12, -114.02), y=slice(46.92, 46.84))
)
# 0 is a WorldCover class, not missing data: drop the GDAL nodata tag
landcover.attrs.pop("_FillValue")
ds = landcover.to_dataset(name="landcover").proj.assign_crs(spatial_ref="EPSG:4326")
ds
print(html_repr(ds))  # markdown-exec: hide
```

Build a pyramid with each method and open the coarsest level (8x coarser than the source). `mask_and_scale=False` keeps the class codes as `uint8`:

```python exec="on" source="above" session="categorical" html="true"
from topozarr import create_pyramid

coarse = {}
for method in ["nearest", "mean"]:
    create_pyramid(ds, levels=4, method=method).write(f"{method}.zarr")
    dt = xr.open_datatree(f"{method}.zarr", engine="zarr", consolidated=False, mask_and_scale=False)
    coarse[method] = dt["3"].landcover

coarse = xr.concat(coarse.values(), dim=xr.Variable("method", list(coarse)))
coarse.to_dataset()
print(html_repr(coarse.to_dataset()))  # markdown-exec: hide
```

`nearest` returns only real classes. `mean` keeps the dtype, but it fills the boundaries between classes with codes that aren't classes (red in the plot below):

```python exec="on" source="above" session="categorical" result="text"
classes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
print((~coarse.isin(classes)).mean(["x", "y"]).to_series())
```

```python exec="on" source="above" session="categorical" html="true"
import carbonplan_styles as cps
from matplotlib.colors import BoundaryNorm, ListedColormap

# one CarbonPlan color per class in `classes`; red marks codes that are not classes
c = cps.colors.light
colors = [c["secondary"], c["green"], c["orange"], c["yellow"], c["pink"], c["purple"],
          c["grey"], cps.colors.dark["text"], c["blue"], c["teal"], c["teal"], c["grey"], c["red"]]
norm = BoundaryNorm([c - 0.5 for c in classes] + [100.5, 255.5], len(colors))
shown = coarse.where(coarse.isin(classes), 255)


def plot(mode):
    grid = shown.plot.imshow(col="method", cmap=ListedColormap(colors), norm=norm, size=5, add_colorbar=False)
    for ax in grid.axs.flat:
        ax.locator_params(nbins=3)
        ax.ticklabel_format(useOffset=False)
    return grid.fig


show(plot)
```

There is a trade-off. `nearest` samples one cell per window, so a rare class that seldom lands on a window's corner can shrink or disappear at coarser levels.

```python exec="on" session="categorical"
# hidden: remove the stores this page wrote
import shutil

for path in ['nearest.zarr', 'mean.zarr']:
    shutil.rmtree(path, ignore_errors=True)
```
