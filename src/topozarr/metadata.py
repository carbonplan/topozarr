from typing import Any
import xarray as xr
from .chunking import (
    calculate_chunk_size,
    calculate_shard_size,
    DEFAULT_CHUNK_BYTES,
    DEFAULT_SHARD_BYTES,
    get_ideal_dim,
)


def create_level_encoding(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    target_shard_bytes: int | None = DEFAULT_SHARD_BYTES,
) -> dict[str, Any]:
    spatial_vars = {
        var_name: da
        for var_name, da in ds.data_vars.items()
        if x_dim in da.dims and y_dim in da.dims
    }

    return {
        var_name: _create_var_encoding(
            da, x_dim, y_dim, target_chunk_bytes, target_shard_bytes
        )
        for var_name, da in spatial_vars.items()
    }


def _create_var_encoding(
    da: xr.DataArray,
    x_dim: str,
    y_dim: str,
    target_chunk_bytes: int,
    target_shard_bytes: int | None,
) -> dict[str, Any]:
    itemsize = da.dtype.itemsize
    ideal_chunk = get_ideal_dim(itemsize, target_chunk_bytes)

    y_idx, x_idx = da.get_axis_num(y_dim), da.get_axis_num(x_dim)

    chunks = list(da.shape)
    shards = list(da.shape) if target_shard_bytes is not None else None

    for idx, dim_name in [(y_idx, y_dim), (x_idx, x_dim)]:
        c = calculate_chunk_size(da.shape[idx], ideal_chunk)
        chunks[idx] = c

        if shards is not None:
            ideal_shard = get_ideal_dim(itemsize, target_shard_bytes)
            shards[idx] = calculate_shard_size(da.shape[idx], c, ideal_shard)

    for i, dim in enumerate(da.dims):
        if dim not in [x_dim, y_dim]:
            chunks[i] = 1
            if shards is not None:
                shards[i] = 1

    var_encoding = {"chunks": tuple(chunks)}
    if shards is not None:
        var_encoding["shards"] = tuple(shards)

    return var_encoding


def _get_affine_transform(ds: xr.Dataset, x_dim: str, y_dim: str) -> list[float]:
    """Extract 6-element affine transform [a, b, c, d, e, f] from coordinate arrays.

    Follows Rasterio/spatial: convention: x = a*col + b*row + c, y = d*col + e*row + f,
    where (0, 0) is the top-left corner of the top-left pixel.
    """
    x = ds[x_dim].values
    y = ds[y_dim].values
    x_res = float(x[1] - x[0]) if len(x) > 1 else 1.0
    y_res = float(y[1] - y[0]) if len(y) > 1 else 1.0
    c = float(x[0]) - 0.5 * x_res  # x-coordinate of top-left pixel corner
    f = float(y[0]) - 0.5 * y_res  # y-coordinate of top-left pixel corner
    return [x_res, 0.0, c, 0.0, y_res, f]


def _get_spatial_bbox(
    ds: xr.Dataset, x_dim: str, y_dim: str, transform: list[float]
) -> list[float]:
    """Compute [xmin, ymin, xmax, ymax] bounding box from the dataset extent."""
    x_res, _, c, _, y_res, f = transform
    nx, ny = ds.sizes[x_dim], ds.sizes[y_dim]
    xmin = min(c, c + x_res * nx)
    xmax = max(c, c + x_res * nx)
    ymin = min(f, f + y_res * ny)
    ymax = max(f, f + y_res * ny)
    return [xmin, ymin, xmax, ymax]


def create_multiscale_metadata(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    level_datasets: dict[int, xr.Dataset],
    crs: str,
    method: str,
) -> dict[str, Any]:
    spatial_dims = {x_dim, y_dim}
    levels = len(level_datasets)
    root_transform = _get_affine_transform(ds, x_dim, y_dim)

    def get_multiscales_transform(level: int) -> dict[str, Any]:
        # scale is relative to derived_from (always 2.0 per coarsening step, not cumulative)
        s = [2.0 if (level > 0 and d in spatial_dims) else 1.0 for d in ds.dims]
        t = [0.5 if (level > 0 and d in spatial_dims) else 0.0 for d in ds.dims]
        return {"scale": s, "translation": t}

    def get_level_spatial_attrs(level: int) -> dict[str, Any]:
        level_ds = level_datasets[level]
        level_transform = _get_affine_transform(level_ds, x_dim, y_dim)
        level_shape = [int(level_ds.sizes[y_dim]), int(level_ds.sizes[x_dim])]
        return {"spatial:transform": level_transform, "spatial:shape": level_shape}

    layout = [
        {
            "asset": str(i),
            "transform": get_multiscales_transform(i),
            **(
                {"derived_from": str(i - 1), "resampling_method": method}
                if i > 0
                else {}
            ),
            **get_level_spatial_attrs(i),
        }
        for i in range(levels)
    ]

    # ref: https://github.com/zarr-conventions/multiscales/blob/main/examples/geospatial-pyramid.json

    return {
        "zarr_conventions": [
            {
                "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
                "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
                "uuid": "d35379db-88df-4056-af3a-620245f8e347",
                "name": "multiscales",
                "description": "Multiscale layout of zarr datasets",
            },
            {
                "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
                "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
                "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
                "name": "proj:",
                "description": "Coordinate reference system information for geospatial data",
            },
            {
                "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
                "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
                "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
                "name": "spatial:",
                "description": "Spatial coordinate information",
            },
        ],
        "multiscales": {"layout": layout, "resampling_method": method},
        "proj:code": crs,
        "spatial:dimensions": [y_dim, x_dim],
        "spatial:transform": root_transform,
        "spatial:bbox": _get_spatial_bbox(ds, x_dim, y_dim, root_transform),
        "spatial:shape": [int(ds.sizes[y_dim]), int(ds.sizes[x_dim])],
    }
