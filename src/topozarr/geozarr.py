from __future__ import annotations

import xarray as xr

from .coarsen import get_crs
from .metadata import ZarrLayerVarConfig, create_geozarr_metadata


def attach_geozarr_metadata(
    ds: xr.Dataset,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    crs: str | None = None,
    layer_hints: dict[str, ZarrLayerVarConfig] | None = None,
) -> xr.Dataset:
    """Return a copy of ``ds`` with geozarr convention attrs (proj + spatial).

    No multiscale pyramid is built; the dataset stays a flat group. Write it
    with ``ds.to_zarr(...)``. ``crs`` defaults to the dataset CRS (xproj).
    """
    if crs is None:
        crs = get_crs(ds)
    attrs = create_geozarr_metadata(ds, x_dim, y_dim, crs, layer_hints)
    out = ds.copy()
    out.attrs = {**ds.attrs, **attrs}
    return out
