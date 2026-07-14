from __future__ import annotations

import numpy as np
import xarray as xr
import xproj  # noqa: F401 - registers .proj accessor

from .chunking import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_CHUNKS_PER_SHARD,
    ChunksPerShard,
    validate_chunks_per_shard,
)
from .metadata import (
    ZarrLayerVarConfig,
    create_level_encoding,
    create_multiscale_metadata,
)
from .pyramid import CoarseningMethod, Pyramid, source_chunks


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


def _coarsen_template(ds: xr.Dataset, x_dim: str, y_dim: str, step: int) -> xr.Dataset:
    """Coarsen the spatial dims of ``ds`` by ``step``: real coarsened coords,
    placeholder data.

    Spatial data variables become zero-cost broadcast arrays of the right
    shape/dtype (filled in by ``Pyramid.write``); non-spatial variables and
    coords pass through unchanged.
    """
    spatial_dims = {x_dim, y_dim}

    coords = {}
    for name, coord in ds.coords.items():
        if coord.ndim == 1 and coord.dims[0] in spatial_dims:
            coords[name] = xr.DataArray(
                _coarsen_coord(coord.values, step), dims=coord.dims, attrs=coord.attrs
            )
        else:
            coords[name] = coord

    data_vars = {}
    for name, da in ds.data_vars.items():
        if spatial_dims <= set(da.dims):
            shape = tuple(
                s // step if d in spatial_dims else s for d, s in zip(da.dims, da.shape)
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
    factors: list[int],
    x_dim: str,
    y_dim: str,
) -> dict[int, xr.Dataset]:
    # mean/max/min/sum are composable, so each level coarsens from the prior
    # (coarser) level by the per-step ratio. NOTE: median/mode are NOT
    # composable -- when added, sparse levels must reduce from native instead.
    levels = [ds]
    for prev_factor, factor in zip(factors[:-1], factors[1:]):
        step = factor // prev_factor
        curr = levels[-1]
        if curr.sizes[x_dim] < step or curr.sizes[y_dim] < step:
            raise ValueError(
                f"cannot coarsen by step {step} (cumulative factor {factor}): "
                f"dimensions are {x_dim}={curr.sizes[x_dim]}, "
                f"{y_dim}={curr.sizes[y_dim]}; both must be >= {step} to coarsen further"
            )
        levels.append(_coarsen_template(curr, x_dim, y_dim, step))

    # zarr-multiscales: level 0 = highest resolution
    return dict(enumerate(levels))


def _resolve_factors(levels: int | None, factors: list[int] | None) -> list[int]:
    """Validate the levels/factors inputs and return cumulative downsample factors.

    Exactly one of ``levels`` / ``factors`` must be given. ``levels=N`` maps to
    powers of two ``[1, 2, 4, ..., 2**(N-1)]``. An explicit ``factors`` list must
    start at 1, be strictly increasing, and have each entry integer-divide the
    next (whole per-step ratios).
    """
    if (levels is None) == (factors is None):
        raise ValueError("pass exactly one of 'levels' or 'factors'")

    if levels is not None:
        if not isinstance(levels, int) or levels < 1:
            raise ValueError(f"levels must be a positive int, got {levels!r}")
        return [2**i for i in range(levels)]

    assert factors is not None
    if not factors:
        raise ValueError("factors must be a non-empty list")
    if any(not isinstance(f, int) or isinstance(f, bool) or f < 1 for f in factors):
        raise ValueError(f"factors must be positive ints, got {factors!r}")
    if factors[0] != 1:
        raise ValueError(f"factors must start at 1, got {factors!r}")
    for prev, curr in zip(factors[:-1], factors[1:]):
        if curr <= prev:
            raise ValueError(f"factors must be strictly increasing, got {factors!r}")
        if curr % prev != 0:
            raise ValueError(
                f"each factor must be an integer multiple of the previous one, "
                f"got {factors!r} ({curr} not divisible by {prev})"
            )
    return list(factors)


def create_pyramid(
    ds: xr.Dataset,
    levels: int | None = None,
    *,
    factors: list[int] | None = None,
    x_dim: str = "x",
    y_dim: str = "y",
    method: CoarseningMethod = "mean",
    target_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunks_per_shard: ChunksPerShard | None = DEFAULT_CHUNKS_PER_SHARD,
    layer_hints: dict[str, ZarrLayerVarConfig] | None = None,
) -> Pyramid:
    """Build a multiscale Zarr pyramid plan from a georeferenced Dataset.

    Exactly one of ``levels`` / ``factors`` must be given.

    Args:
        ds: Source dataset. Must have a CRS assigned via ``ds.proj.assign_crs``.
        levels: Total number of resolution levels, including the original.
            Level ``0`` is the original resolution; each subsequent level
            coarsens by 2× per spatial dimension (cumulative factors
            ``[1, 2, 4, ...]``).
        factors: Explicit cumulative downsample factors per level, e.g.
            ``[1, 4, 16]`` for a sparse 4×-spaced pyramid. Must start at 1, be
            strictly increasing, and have each entry integer-divide the next.
            Mutually exclusive with ``levels``.
        x_dim: Name of the x (longitude / easting) dimension.
        y_dim: Name of the y (latitude / northing) dimension.
        method: Spatial aggregation method for coarsening. Integer variables
            keep their dtype: ``mean`` truncates toward zero (unlike
            ``xarray.coarsen``, which promotes to float).
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

        # sparse pyramid: native, 4x, 16x (skips the costly 2x level)
        sparse = create_pyramid(ds, factors=[1, 4, 16], x_dim="lon", y_dim="lat")
        sparse.write("sparse.zarr")
        ```
    """
    factors = _resolve_factors(levels, factors)
    if chunks_per_shard is not None:
        validate_chunks_per_shard(chunks_per_shard)
    for name, da in ds.data_vars.items():
        if x_dim in da.dims and y_dim in da.dims and da.ndim > 4:
            raise ValueError(
                f"spatial variable {name!r} has {da.ndim} dimensions; "
                "topozarr-core supports at most 4 "
                "(use pyramid.as_datatree() for the xarray/Dask path, which lifts this limit)"
            )
    crs_str = get_crs(ds)
    level_templates = build_level_templates(ds, factors, x_dim, y_dim)

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
        factors=factors,
        layer_hints=layer_hints,
    )

    # _FillValue if declared; else NaN for floats (matches xarray's zarr
    # default and lets the engine skip all-fill regions)
    fill_values = {
        str(name): da.encoding.get(
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
        factors=factors,
        fill_values=fill_values,
    )
