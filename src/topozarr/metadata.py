from typing import Any, Literal
import xarray as xr
from .chunking import (
    calculate_chunk_size,
    calculate_shard_size,
    DEFAULT_CHUNK_BYTES,
    DEFAULT_SHARD_BYTES,
    get_ideal_dim,
)


SpecType = Literal["ndpyramid", "zarr-multiscales"]


def create_level_encoding(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    target_shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> dict[str, Any]:
    encoding = {}
    for var_name, da in ds.data_vars.items():
        if x_dim not in da.dims or y_dim not in da.dims:
            continue

        itemsize = da.dtype.itemsize
        ideal_chunk = get_ideal_dim(itemsize, target_chunk_bytes)
        ideal_shard = get_ideal_dim(itemsize, target_shard_bytes)

        y_idx, x_idx = da.get_axis_num(y_dim), da.get_axis_num(x_dim)

        chunks = list(da.shape)
        shards = list(da.shape)

        for idx, dim_name in [(y_idx, y_dim), (x_idx, x_dim)]:
            c = calculate_chunk_size(da.shape[idx], ideal_chunk)
            chunks[idx] = c
            shards[idx] = calculate_shard_size(da.shape[idx], c, ideal_shard)

        for i, dim in enumerate(da.dims):
            if dim not in [x_dim, y_dim]:
                chunks[i] = 1
                shards[i] = 1

        encoding[var_name] = {"chunks": tuple(chunks), "shards": tuple(shards)}
    return encoding


def create_multiscale_metadata(
    levels: int, crs: str, method: str, spec: SpecType = "ndpyramid"
) -> dict[str, Any]:
    indices = list(range(levels))

    if spec == "ndpyramid":
        # ndpyramid-ish (highest levels = highest resolution)
        datasets = [{"path": str(i), "level": i, "crs": crs} for i in reversed(indices)]
        return {
            "multiscales": [
                {
                    "datasets": datasets,
                    "type": "reduce",
                    "metadata": {"method": "coarsen", "coarsening_method": method},
                }
            ]
        }

    # zarr-multiscales (as of Jan 8th 2026) (lowest levels = highest resolution)
    layout = [
        {
            "asset": str(i),
            "transform": {
                "scale": [float(2**i), float(2**i)],
                "translation": [0.0, 0.0],
            },
            **({"derived_from": str(i - 1)} if i > 0 else {}),
        }
        for i in indices
    ]

    return {
        "multiscales": {"layout": layout, "resampling_method": method},
        "proj:code": crs,
    }
