from __future__ import annotations

import numpy as np
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor

from .metadata import (
    create_level_encoding,
    create_multiscale_metadata,
    ZarrLayerVarConfig,
)
from .pyramid import CoarseningMethod, Pyramid, source_chunks
from .chunking import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CHUNKS_PER_SHARD,
    ChunksPerShard,
    validate_chunks_per_shard,
)


def get_crs(ds: xr.Dataset) -> str:
    crs = ds.proj.crs
    if not crs:
        raise ValueError(
            "dataset is missing a crs. Assign one with xproj, "
            'e.g. ds.proj.assign_crs(spatial_ref="EPSG:4326").'
        )
    return str(crs)


def _coarsen_coord(values: np.ndarray, factor: int) -> np.ndarray:
    """Mean of cell-center coordinates per window, trimming trailing partials."""
    n = max(len(values) // factor, 1)
    window = min(factor, len(values))
    return values[: n * window].reshape(n, window).mean(axis=1)


def _coarsen_template(ds: xr.Dataset, x_dim: str, y_dim: str) -> xr.Dataset:
    """Halve the spatial dims of ``ds``: real coarsened coords, placeholder data.

    Spatial data variables become zero-cost broadcast arrays of the right
    shape/dtype (filled in by ``Pyramid.write``); non-spatial variables and
    coords pass through unchanged.
    """
    spatial_dims = {x_dim, y_dim}

    coords = {}
    for name, coord in ds.coords.items():
        if coord.ndim == 1 and coord.dims[0] in spatial_dims:
            coords[name] = xr.DataArray(
                _coarsen_coord(coord.values, 2), dims=coord.dims, attrs=coord.attrs
            )
        else:
            coords[name] = coord

    data_vars = {}
    for name, da in ds.data_vars.items():
        if spatial_dims <= set(da.dims):
            shape = tuple(
                s // 2 if d in spatial_dims else s for d, s in zip(da.dims, da.shape)
            )
            placeholder = np.broadcast_to(np.zeros(1, dtype=da.dtype), shape)
            data_vars[name] = xr.DataArray(placeholder, dims=da.dims, attrs=da.attrs)
        else:
            data_vars[name] = da

    return xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)


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


def build_level_templates(
    ds: xr.Dataset,
    num_levels: int,
    x_dim: str,
    y_dim: str,
) -> dict[int, xr.Dataset]:
    levels = [ds]
    for lvl in range(num_levels - 1):
        curr = levels[-1]
        if curr.sizes[x_dim] < 2 or curr.sizes[y_dim] < 2:
            raise ValueError(
                f"cannot coarsen to {num_levels} levels: after {lvl} step(s) "
                f"dimensions are {x_dim}={curr.sizes[x_dim]}, {y_dim}={curr.sizes[y_dim]}; "
                "both must be >= 2 to coarsen further"
            )
        levels.append(_coarsen_template(curr, x_dim, y_dim))

    # zarr-multiscales: level 0 = highest resolution
    return dict(enumerate(levels))


def create_pyramid(
    ds: xr.Dataset,
    levels: int,
    x_dim: str = "x",
    y_dim: str = "y",
    method: CoarseningMethod = "mean",
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunks_per_shard: ChunksPerShard | None = DEFAULT_CHUNKS_PER_SHARD,
    layer_hints: dict[str, ZarrLayerVarConfig] | None = None,
) -> Pyramid:
    """Build a multiscale Zarr pyramid plan from a georeferenced Dataset.

    Args:
        ds: Source dataset. Must have a CRS assigned via ``ds.proj.assign_crs``.
        levels: Total number of resolution levels, including the original.
            Level ``0`` is the original resolution; each subsequent level
            coarsens by 2× per spatial dimension.
        x_dim: Name of the x (longitude / easting) dimension.
        y_dim: Name of the y (latitude / northing) dimension.
        method: Spatial aggregation method for coarsening.
        target_chunk_bytes: Target uncompressed size per chunk (default ~500 KB).
        chunks_per_shard: Number of chunks per shard along each spatial dimension
            (e.g. ``4`` → 4×4 = 16 chunks per shard, ~8 MB). Must be a power
            of 2 in the range 1–32. Pass ``None`` to disable sharding.
        layer_hints: Optional per-variable colormap / color-range hints written
            into the ``zarr-layer`` root metadata key.

    Returns:
        A [Pyramid][topozarr.pyramid.Pyramid] write plan; call
        ``pyramid.write(store)`` to compute and write all levels.

    Raises:
        ValueError: If ``ds`` has no CRS, ``chunks_per_shard`` is not a
            power of 2 in the range 1–32, or a spatial variable has more
            than 4 dimensions (topozarr-core kernel limit).

    Examples:
        ```python
        import xarray as xr
        import xproj  # registers the .proj accessor
        from topozarr import create_pyramid

        ds = xr.tutorial.open_dataset("air_temperature").drop_encoding()
        ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

        pyramid = create_pyramid(ds, levels=2, x_dim="lon", y_dim="lat")
        pyramid.write("pyramid.zarr")
        ```
    """
    if chunks_per_shard is not None:
        validate_chunks_per_shard(chunks_per_shard)
    for name, da in ds.data_vars.items():
        if x_dim in da.dims and y_dim in da.dims and da.ndim > 4:
            raise ValueError(
                f"spatial variable {name!r} has {da.ndim} dimensions; "
                "the native engine (topozarr-core) supports at most 4 "
                "(use engine='xarray' in pyramid.write() to lift this limit)"
            )
    crs_str = get_crs(ds)
    level_templates = build_level_templates(ds, levels, x_dim, y_dim)

    level0_source_chunks = _spatial_source_chunks(ds, x_dim, y_dim)
    full_encoding = {
        f"/{idx}": create_level_encoding(
            template,
            x_dim,
            y_dim,
            target_chunk_bytes=target_chunk_bytes,
            chunks_per_shard=chunks_per_shard,
            source_chunks=level0_source_chunks if idx == 0 else None,
        )
        for idx, template in level_templates.items()
    }

    attrs = create_multiscale_metadata(
        ds=ds,
        x_dim=x_dim,
        y_dim=y_dim,
        level_datasets=level_templates,
        crs=crs_str,
        method=str(method),
        layer_hints=layer_hints,
    )

    # _FillValue if declared; else NaN for floats (matches xarray's zarr
    # default and lets the engine skip all-fill regions)
    fill_values = {
        name: da.encoding.get(
            "_FillValue",
            da.attrs.get(
                "_FillValue",
                np.nan if np.issubdtype(da.dtype, np.floating) else None,
            ),
        )
        for name, da in ds.data_vars.items()
    }

    return Pyramid(
        source=ds,
        level_templates=level_templates,
        encoding=full_encoding,
        attrs=attrs,
        x_dim=x_dim,
        y_dim=y_dim,
        method=method,
        fill_values=fill_values,
    )
