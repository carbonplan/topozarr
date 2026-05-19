from __future__ import annotations
from typing import Any
import zarr
import xarray as xr


def _level_store(store: Any, level: int) -> tuple[Any, str | None]:
    """Return (store, group) for accessing a pyramid level subgroup.

    zarr v3 raises ValueError if a path/group argument is passed alongside an
    FSMap (fsspec) store — the subgroup path must be embedded in the mapper
    instead.  For all other store types the caller can pass group= as usual.
    """
    try:
        from fsspec.mapping import FSMap
        if isinstance(store, FSMap):
            root = store.root.rstrip("/")
            return store.fs.get_mapper(f"{root}/{level}"), None
    except ImportError:
        pass
    return store, str(level)


def write_pyramid(
    pyramid: Any,
    store: Any,
    x_dim: str = "x",
    y_dim: str = "y",
    method: str = "mean",
    mode: str = "w",
    zarr_format: int = 3,
    consolidated: bool = False,
) -> None:
    """Write a pyramid to a zarr store one level at a time, breaking the dask chain.

    ``pyramid.dt.to_zarr()`` submits all pyramid levels as a single dask graph.
    Because each coarser level's graph chains back through every finer level,
    dask may hold many intermediate full-resolution arrays in memory simultaneously,
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
        Destination zarr store, FSMap, or path string.
    x_dim:
        Name of the x spatial dimension — must match what was passed to ``create_pyramid``.
    y_dim:
        Name of the y spatial dimension — must match what was passed to ``create_pyramid``.
    method:
        Coarsening method — must match what was passed to ``create_pyramid``.
    mode:
        ``'w'`` to overwrite, ``'a'`` to append/update.
    zarr_format:
        Zarr format version (2 or 3).
    consolidated:
        Write consolidated metadata after all levels are written.
        Ignored for zarr v3 (consolidated metadata is not part of the spec).
    """
    n_levels = len(pyramid.encoding)

    # Write root group and its multiscale/spatial attributes into zarr.json.
    # zarr v3 accepts FSMap here because no subgroup path is involved.
    root = zarr.open_group(store, mode=mode, zarr_format=zarr_format)
    root.attrs.update(pyramid.dt.attrs)

    # Level 0 is the finest resolution — write directly from the pyramid DataTree.
    level_store, group = _level_store(store, 0)
    pyramid.dt["/0"].ds.to_zarr(
        level_store,
        group=group,
        mode="a",
        encoding=pyramid.encoding["/0"],
        zarr_format=zarr_format,
        consolidated=False,
    )

    # For each subsequent level, read the previous level back from zarr to get
    # fresh dask arrays (no chained dependency on level 0's full-resolution data),
    # then coarsen by 2x and write.
    for i in range(1, n_levels):
        enc = pyramid.encoding[f"/{i}"]
        template_ds = pyramid.dt[f"/{i}"].ds

        # Opening from zarr breaks the dask graph chain.
        prev_store, prev_group = _level_store(store, i - 1)
        prev_ds = xr.open_zarr(prev_store, group=prev_group, consolidated=False)

        curr_ds = getattr(
            prev_ds.coarsen({x_dim: 2, y_dim: 2}, boundary="trim"), method
        )()

        # coarsen drops dataset and variable attributes; restore from template.
        curr_ds.attrs = template_ds.attrs
        for var in curr_ds.data_vars:
            if var in template_ds:
                curr_ds[var].attrs = template_ds[var].attrs

        curr_store, curr_group = _level_store(store, i)
        curr_ds.to_zarr(
            curr_store,
            group=curr_group,
            mode="a",
            encoding=enc,
            zarr_format=zarr_format,
            consolidated=False,
        )

    if consolidated and zarr_format != 3:
        zarr.consolidate_metadata(store)
