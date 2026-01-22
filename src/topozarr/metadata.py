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


def create_multiscale_metadata(
    levels: int,
    crs: str,
    method: str,
) -> dict[str, Any]:
    layout = []

    for i in range(levels):
        entry = {
            "asset": str(i),
            "transform": {
                "scale": [float(2**i), float(2**i)],
                "translation": [0.5, 0.5] if i > 0 else [0.0, 0.0],
            },
        }

        if i > 0:
            entry["derived_from"] = str(i - 1)
            entry["resampling_method"] = method

        layout.append(entry)

    # attempting to match this example: https://github.com/zarr-conventions/multiscales/blob/main/examples/array-based-pyramid.json
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
        ],
        "multiscales": {"layout": layout, "resampling_method": method},
        "proj:code": crs,
    }
