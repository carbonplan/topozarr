from __future__ import annotations
from typing import Any
import zarr
import xarray as xr
from .pyramid import Pyramid


def _rechunk_to_shards(ds: xr.Dataset, level_enc: dict[str, Any]) -> xr.Dataset:
    """Rechunk dataset so each dask chunk aligns with one shard for efficient writes."""
    dim_chunks: dict[str, int] = {}
    for var_name, var_enc in level_enc.items():
        if var_name not in ds.data_vars or "chunks" not in var_enc:
            continue
        write_chunks = var_enc.get("shards", var_enc["chunks"])
        da = ds[var_name]
        for dim, size in zip(da.dims, write_chunks):
            if dim not in dim_chunks:
                dim_chunks[dim] = size
            else:
                dim_chunks[dim] = min(dim_chunks[dim], size)
    if dim_chunks:
        ds = ds.chunk(dim_chunks)
    return ds


def write_pyramid(
    pyramid: Pyramid,
    store: Any,
    mode: str = "w",
    zarr_format: int = 3,
    consolidated: bool = False,
) -> None:
    """Write a pyramid to a zarr store one level at a time, breaking the dask chain.

    ``pyramid.dt.to_zarr()`` submits all pyramid levels as a single dask graph.
    Because each coarser level's graph chains back through every finer level, dask
    may hold many intermediate full-resolution arrays in memory simultaneously,
    causing OOM on large datasets.

    This function avoids that by:
    1. Writing level 0 (finest) directly from the pyramid's DataTree.
    2. For each subsequent level, reading the just-written zarr group back as fresh
       dask arrays (no inherited graph) and coarsening from there.

    Parameters
    ----------
    pyramid:
        Pyramid created by ``create_pyramid``.
    store:
        Destination zarr store or path string.
    mode:
        ``'w'`` to overwrite, ``'a'`` to append/update.
    zarr_format:
        Zarr format version (2 or 3).
    consolidated:
        Write consolidated metadata after all levels are written.
        Ignored for zarr v3 (consolidated metadata is not part of the spec).
    """
    n_levels = len(pyramid.encoding)

    # Write root group and its multiscale/spatial attributes.
    root = zarr.open_group(store, mode=mode, zarr_format=zarr_format)
    root.attrs.update(pyramid.dt.attrs)

    # Level 0 is the finest resolution — write directly from the pyramid DataTree.
    level_0_enc = pyramid.encoding["/0"]
    level_0_ds = _rechunk_to_shards(pyramid.dt["/0"].ds, level_0_enc)
    level_0_ds.to_zarr(
        store,
        group="0",
        mode="a",
        encoding=level_0_enc,
        zarr_format=zarr_format,
        consolidated=False,
    )

    # For each subsequent level, read the previous level back from zarr to get
    # fresh dask arrays (no chained dependency on level 0's full-resolution data),
    # then coarsen by 2x and write.
    for i in range(1, n_levels):
        curr_path = f"/{i}"
        curr_enc = pyramid.encoding[curr_path]
        template_ds = pyramid.dt[curr_path].ds

        # Opening from zarr breaks the dask graph chain: the resulting arrays
        # depend only on the on-disk bytes, not on the previous coarsen graph.
        prev_ds = xr.open_zarr(store, group=str(i - 1), consolidated=False)

        coarsened = prev_ds.coarsen(
            {pyramid.x_dim: 2, pyramid.y_dim: 2}, boundary="trim"
        )
        curr_ds = getattr(coarsened, pyramid.method)()

        # coarsen drops dataset and variable attributes; restore from template.
        curr_ds.attrs = template_ds.attrs
        for var in curr_ds.data_vars:
            if var in template_ds:
                curr_ds[var].attrs = template_ds[var].attrs

        curr_ds = _rechunk_to_shards(curr_ds, curr_enc)
        curr_ds.to_zarr(
            store,
            group=str(i),
            mode="a",
            encoding=curr_enc,
            zarr_format=zarr_format,
            consolidated=False,
        )

    if consolidated and zarr_format != 3:
        zarr.consolidate_metadata(store)
