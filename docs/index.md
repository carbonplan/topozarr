---
hide:
  - toc
---

# topozarr { .tz-hidden }

<div class="tz-hero" markdown>

![topozarr](assets/topozarr-lockup-on-light.svg#only-light){ .tz-lockup }
![topozarr](assets/topozarr-lockup-on-dark.svg#only-dark){ .tz-lockup }

<p class="tz-tagline">Create multiscale Zarr stores for web visualization.</p>

</div>

Built for use with [zarr-layer](https://zarr-layer.demo.carbonplan.org/). Follows the [zarr-conventions](https://github.com/zarr-conventions):

- [multiscales](https://github.com/zarr-conventions/multiscales) — pyramid structure and resolution levels
- [proj:](https://github.com/zarr-conventions/proj) — coordinate reference system (CRS)
- [spatial:](https://github.com/zarr-conventions/spatial) — affine transform, bounding box, and dimension names

## Installation

=== "uv"

    ```bash
    uv add topozarr

    # optional extras
    uv add 'topozarr[tutorial]'  # deps for the Usage examples
    uv add 'topozarr[zarrs]'     # faster local writes (Rust codec pipeline)
    ```

=== "pip"

    ```bash
    pip install topozarr

    # optional extras
    pip install 'topozarr[tutorial]'  # deps for the Usage examples
    pip install 'topozarr[zarrs]'     # faster local writes (Rust codec pipeline)
    ```

=== "pixi"

    ```bash
    pixi add --pypi topozarr

    # optional extras
    pixi add --pypi 'topozarr[tutorial]'  # deps for the Usage examples
    pixi add --pypi 'topozarr[zarrs]'     # faster local writes (Rust codec pipeline)
    ```

## Quick example

```python
import xarray as xr
import xproj  # for CRS assignment
from topozarr import create_pyramid

ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
pyramid.write("pyramid.zarr")
```

Open the result in [zarr-layer](https://zarr-layer.demo.carbonplan.org/). See [Usage](usage.md) for cloud stores and options, and [Tips & caveats](tips.md) for performance and gotchas.
