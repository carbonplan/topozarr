from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor
from pyproj import CRS

from .chunking import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CHUNKS_PER_SHARD,
    ChunksPerShard,
    calculate_chunk_size,
    calculate_shard_size,
    fill_nonspatial_shards,
    get_ideal_dim,
    snap_chunk_to_source,
    source_chunks,
    validate_chunks_per_shard,
)

MULTISCALES_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
    "name": "multiscales",
    "description": "Multiscale layout of zarr datasets",
}

PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data",
}

SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial",
    "description": "Spatial coordinate information",
}


def get_crs(ds: xr.Dataset) -> str:
    crs = ds.proj.crs
    if not crs:
        raise ValueError(
            "dataset is missing a crs. Assign one with xproj, "
            'e.g. ds.proj.assign_crs(spatial_ref="EPSG:4326").'
        )
    return str(crs)


@dataclass
class ZarrLayerVarConfig:
    """Per-variable visualization hints for zarr-layer.

    Attributes:
        clim: Color range as ``[min, max]``.
        colormap: Colormap name (e.g. ``"blues"``).

    Examples:
        ```python
        create_pyramid(
            ds,
            levels=2,
            layer_hints={"air": ZarrLayerVarConfig(colormap="blues", clim=[230, 310])},
        )
        ```
    """

    clim: list[float] | None = None
    colormap: str | None = None


def create_level_encoding(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunks_per_shard: ChunksPerShard | None = DEFAULT_CHUNKS_PER_SHARD,
    source_chunks: dict[str, int] | None = None,
) -> dict[str, Any]:
    spatial_vars = {
        str(var_name): da
        for var_name, da in ds.data_vars.items()
        if x_dim in da.dims and y_dim in da.dims
    }

    return {
        var_name: _create_var_encoding(
            da, x_dim, y_dim, target_chunk_bytes, chunks_per_shard, source_chunks
        )
        for var_name, da in spatial_vars.items()
    }


def _create_var_encoding(
    da: xr.DataArray,
    x_dim: str,
    y_dim: str,
    target_chunk_bytes: int,
    chunks_per_shard: ChunksPerShard | None,
    source_chunks: dict[str, int] | None = None,
) -> dict[str, Any]:
    itemsize = da.dtype.itemsize
    ideal_chunk = get_ideal_dim(itemsize, target_chunk_bytes)

    y_idx, x_idx = da.get_axis_num(y_dim), da.get_axis_num(x_dim)

    chunks = list(da.shape)
    shards = list(da.shape) if chunks_per_shard is not None else None

    for idx, dim_name in [(y_idx, y_dim), (x_idx, x_dim)]:
        c = None
        if source_chunks is not None and dim_name in source_chunks:
            c = snap_chunk_to_source(
                da.shape[idx], ideal_chunk, source_chunks[dim_name], chunks_per_shard
            )
        if c is None:
            c = calculate_chunk_size(da.shape[idx], ideal_chunk)
        chunks[idx] = c

        if shards is not None and chunks_per_shard is not None:
            shards[idx] = calculate_shard_size(da.shape[idx], c, chunks_per_shard)

    nonspatial_idx = []
    for i, dim in enumerate(da.dims):
        if dim not in [x_dim, y_dim]:
            nonspatial_idx.append(i)
            chunks[i] = 1
            if shards is not None:
                shards[i] = 1

    if shards is not None and chunks_per_shard is not None:
        shards = fill_nonspatial_shards(
            shards,
            chunks,
            da.shape,
            nonspatial_idx,
            itemsize,
            target_chunk_bytes,
            chunks_per_shard,
        )

    var_encoding = {"chunks": tuple(chunks)}
    if shards is not None:
        var_encoding["shards"] = tuple(shards)

    return var_encoding


def _spatial_source_chunks(
    ds: xr.Dataset, x_dim: str, y_dim: str
) -> dict[str, int] | None:
    """Source chunk size per spatial dim, if all spatial vars agree."""
    sizes: dict[str, set[int]] = {x_dim: set(), y_dim: set()}
    for da in ds.data_vars.values():
        if not {x_dim, y_dim} <= set(da.dims):
            continue
        chunks = source_chunks(da)
        if chunks is None:
            return None
        for dim in (x_dim, y_dim):
            sizes[dim].add(chunks[da.get_axis_num(dim)])
    if any(len(s) != 1 for s in sizes.values()):
        return None
    return {dim: s.pop() for dim, s in sizes.items()}


def recommend_encoding(
    ds: xr.Dataset,
    *,
    x_dim: str = "x",
    y_dim: str = "y",
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunks_per_shard: ChunksPerShard | None = DEFAULT_CHUNKS_PER_SHARD,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Recommended ``chunks`` / ``shards`` for writing ``ds`` as a flat group.

    Same heuristic `create_pyramid` applies per level, exposed for
    single-resolution datasets. Pass the result straight to
    ``ds.to_zarr(..., encoding=...)``. No CRS is required — the encoding
    depends only on shape and dtype.

    Args:
        ds: Source dataset. Chunking of a chunked source (dask or a zarr
            backend) is sniffed and the recommendation snapped to nest with it.
        x_dim: Name of the x (longitude / easting) dimension.
        y_dim: Name of the y (latitude / northing) dimension.
        target_chunk_bytes: Target uncompressed size per chunk (default ~500 KB).
        chunks_per_shard: Number of chunks per shard along each spatial
            dimension (e.g. ``4`` → 4×4 = 16 chunks per shard, ~8 MB). Must be a
            power of 2 in the range 1–32. Pass ``None`` to disable sharding.

            This also sets the shard byte budget: whatever the spatial dims
            leave unused widens non-spatial dims such as ``time`` or ``band``.
            Chunks along those dims stay at 1.

    Returns:
        ``{var_name: {"chunks": (...), "shards": (...)}}`` for variables that
        have both spatial dims; ``"shards"`` is omitted when
        ``chunks_per_shard`` is ``None``. Non-spatial variables are absent and
        fall through to xarray's defaults.

    Raises:
        ValueError: If ``chunks_per_shard`` is not a power of 2 in the range
            1–32, ``x_dim`` or ``y_dim`` is not a dimension of ``ds``, or no
            data variable has both spatial dimensions.

    Examples:
        ```python
        from topozarr import attach_geozarr_metadata, recommend_encoding

        ds = attach_geozarr_metadata(ds, x_dim="lon", y_dim="lat")
        ds.to_zarr(
            "flat.zarr",
            zarr_format=3,
            consolidated=False,
            encoding=recommend_encoding(ds, x_dim="lon", y_dim="lat"),
        )
        ```

    Note:
        For a dask-backed ``ds``, xarray's ``safe_chunks`` check requires dask
        blocks to align with the zarr write unit -- the *shard*, when sharding
        is on. Only spatial chunks are snapped to the source, not shards, so a
        dask source usually fails this check. Rechunk to the recommended shards
        before writing, or pass ``safe_chunks=False``. A lazily opened
        zarr-backed ``ds`` (``chunks=None``) is unaffected.
    """
    if chunks_per_shard is not None:
        validate_chunks_per_shard(chunks_per_shard)
    if x_dim not in ds.dims:
        raise ValueError(f"x_dim {x_dim!r} not found in dataset dims {tuple(ds.dims)}")
    if y_dim not in ds.dims:
        raise ValueError(f"y_dim {y_dim!r} not found in dataset dims {tuple(ds.dims)}")
    if not any(x_dim in da.dims and y_dim in da.dims for da in ds.data_vars.values()):
        raise ValueError(
            f"no variable has both x_dim {x_dim!r} and y_dim {y_dim!r}; "
            "nothing to encode"
        )

    return create_level_encoding(
        ds,
        x_dim,
        y_dim,
        target_chunk_bytes=target_chunk_bytes,
        chunks_per_shard=chunks_per_shard,
        source_chunks=_spatial_source_chunks(ds, x_dim, y_dim),
    )


def _coord_resolution(values: np.ndarray, dim: str, fallback: float | None) -> float:
    """Spacing of a 1-D coordinate array; raises on non-uniform spacing.

    A single-element coordinate (a fully coarsened dimension) carries no
    spacing information, so ``fallback`` (derived from the level-0 resolution)
    is required in that case.
    """
    if len(values) > 1:
        diffs = np.diff(values.astype("float64"))
        if not np.allclose(diffs, diffs[0], rtol=1e-5):
            raise ValueError(
                f"coordinate {dim!r} is not uniformly spaced; "
                "topozarr requires a regular grid"
            )
        return float(diffs[0])
    if fallback is None:
        raise ValueError(
            f"cannot infer resolution of coordinate {dim!r} from a single value"
        )
    return fallback


def _get_affine_transform(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    fallback_res: tuple[float, float] | None = None,
) -> list[float]:
    """Extract 6-element affine transform [a, b, c, d, e, f] from coordinate arrays.

    Follows Rasterio/spatial: convention: x = a*col + b*row + c, y = d*col + e*row + f,
    where (0, 0) is the top-left corner of the top-left pixel.

    Coordinates must be uniformly spaced. ``fallback_res`` supplies the
    (x, y) resolution for length-1 dimensions, which have no measurable
    spacing of their own.
    """
    x = ds[x_dim].values
    y = ds[y_dim].values
    x_res = _coord_resolution(x, x_dim, fallback_res[0] if fallback_res else None)
    y_res = _coord_resolution(y, y_dim, fallback_res[1] if fallback_res else None)
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


def _geozarr_attrs(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    crs: str,
    transform: list[float],
    conventions: list[dict[str, str]],
    layer_hints: dict[str, ZarrLayerVarConfig] | None,
) -> dict[str, Any]:
    """proj + spatial attr block, shared by the flat and multiscale roots."""
    return {
        "zarr_conventions": conventions,
        "proj:code": crs,
        "proj:wkt2": CRS.from_user_input(crs).to_wkt(),
        "spatial:dimensions": [y_dim, x_dim],
        "spatial:registration": "pixel",
        "spatial:transform": transform,
        "spatial:bbox": _get_spatial_bbox(ds, x_dim, y_dim, transform),
        "spatial:shape": [int(ds.sizes[y_dim]), int(ds.sizes[x_dim])],
        **(
            {"zarr-layer": {k: asdict(v) for k, v in layer_hints.items()}}
            if layer_hints is not None
            else {}
        ),
    }


def create_geozarr_metadata(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    crs: str,
    layer_hints: dict[str, ZarrLayerVarConfig] | None = None,
) -> dict[str, Any]:
    """Geozarr convention attrs (proj + spatial) for a single-level dataset.

    No ``multiscales`` convention or layout; suitable for a flat zarr group.
    """
    return _geozarr_attrs(
        ds,
        x_dim,
        y_dim,
        crs,
        _get_affine_transform(ds, x_dim, y_dim),
        [PROJ_CONVENTION, SPATIAL_CONVENTION],
        layer_hints,
    )


def create_multiscale_metadata(
    ds: xr.Dataset,
    x_dim: str,
    y_dim: str,
    level_datasets: dict[int, xr.Dataset],
    crs: str,
    method: str,
    factors: list[int],
    layer_hints: dict[str, ZarrLayerVarConfig] | None = None,
) -> dict[str, Any]:
    spatial_dims = {x_dim, y_dim}
    levels = len(level_datasets)
    root_transform = _get_affine_transform(ds, x_dim, y_dim)

    def get_multiscales_transform(level: int) -> dict[str, Any]:
        # scale is relative to derived_from: the per-step ratio, not cumulative.
        # translation is the pixel-registration half-cell shift ((step-1)/2;
        # step=2 -> 0.5).
        step = factors[level] // factors[level - 1]
        s = [float(step) if (level > 0 and d in spatial_dims) else 1.0 for d in ds.dims]
        t = [
            (step - 1) / 2 if (level > 0 and d in spatial_dims) else 0.0
            for d in ds.dims
        ]
        return {"scale": s, "translation": t}

    def get_level_spatial_attrs(level: int) -> dict[str, Any]:
        level_ds = level_datasets[level]
        # real coords drive the transform for multi-element dims; length-1 dims
        # fall back to level-0 resolution * cumulative factor for this level
        fallback = (
            root_transform[0] * factors[level],
            root_transform[4] * factors[level],
        )
        level_transform = _get_affine_transform(
            level_ds, x_dim, y_dim, fallback_res=fallback
        )
        level_shape = [int(level_ds.sizes[y_dim]), int(level_ds.sizes[x_dim])]
        return {"spatial:transform": level_transform, "spatial:shape": level_shape}

    layout = [
        {
            "asset": str(i),
            **(
                {
                    "derived_from": str(i - 1),
                    "transform": get_multiscales_transform(i),
                    "resampling_method": method,
                }
                if i > 0
                else {}
            ),
            **get_level_spatial_attrs(i),
        }
        for i in range(levels)
    ]

    # ref: https://github.com/zarr-conventions/multiscales/blob/main/examples/geospatial-pyramid.json

    attrs = _geozarr_attrs(
        ds,
        x_dim,
        y_dim,
        crs,
        root_transform,
        [MULTISCALES_CONVENTION, PROJ_CONVENTION, SPATIAL_CONVENTION],
        layer_hints,
    )
    attrs["multiscales"] = {"layout": layout, "resampling_method": method}
    return attrs
