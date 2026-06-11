from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr
import zarr

from .engine import (
    DEFAULT_MAX_REGION_BYTES,
    copy_array,
    copy_region_shape,
    default_max_workers,
    downsample_level,
)

CoarseningMethod = Literal["mean", "max", "min", "sum"]


def _progress_bar(total: int) -> Any:
    try:
        from tqdm.auto import tqdm
    except ImportError as err:
        raise ImportError(
            "progress=True requires tqdm; install it with `pip install tqdm`"
        ) from err
    return tqdm(total=total, unit="region")


def source_chunks(da: xr.DataArray) -> tuple[int, ...] | None:
    """Per-axis chunk shape of the source backing ``da``, if chunked."""
    if da.chunks is not None:  # dask
        return tuple(c[0] for c in da.chunks)
    enc = da.encoding.get("chunks")  # zarr/icechunk backend
    return tuple(enc) if enc is not None else None


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@dataclass
class Pyramid:
    """A write plan for a multiscale Zarr pyramid, returned by
    [create_pyramid][topozarr.coarsen.create_pyramid].

    Attributes:
        source: The original (level 0) dataset.
        level_templates: Per-level datasets carrying real coordinates and
            attrs; spatial data variables are zero-cost placeholders with the
            correct shape/dtype (their data is computed during
            [write][topozarr.pyramid.Pyramid.write]).
        encoding: Nested dict ``{path: {var: {"chunks": ..., "shards": ...}}}``.
        attrs: Root group metadata (multiscales / proj: / spatial: / zarr-layer).
    """

    source: xr.Dataset
    level_templates: dict[int, xr.Dataset]
    encoding: dict[str, Any]
    attrs: dict[str, Any]
    x_dim: str
    y_dim: str
    method: CoarseningMethod
    fill_values: dict[str, float | int | None] = field(default_factory=dict)

    @property
    def levels(self) -> int:
        return len(self.level_templates)

    def _spatial_vars(self) -> list[str]:
        return [
            name
            for name, da in self.source.data_vars.items()
            if self.x_dim in da.dims and self.y_dim in da.dims
        ]

    def _region_shape(
        self, lvl: int, name: str, max_region_bytes: int
    ) -> tuple[int, ...]:
        """Region shape used to stream one variable of one level."""
        template_da = self.level_templates[lvl][name]
        enc = self.encoding[f"/{lvl}"][name]
        shard = tuple(enc.get("shards") or enc["chunks"])
        if lvl > 0:
            return shard
        return copy_region_shape(
            shard,
            template_da.shape,
            template_da.dtype.itemsize,
            source_chunks(self.source[name]),
            max_region_bytes,
        )

    def _region_bytes(self, lvl: int, name: str, max_region_bytes: int) -> int:
        """Approximate bytes held in memory per in-flight region."""
        template_da = self.level_templates[lvl][name]
        region = self._region_shape(lvl, name, max_region_bytes)
        nbytes = math.prod(region) * template_da.dtype.itemsize
        if lvl > 0:
            # the input block is the output region scaled by the 2x2 stride
            nbytes *= 4
        return nbytes

    def _region_count(self, lvl: int, name: str, max_region_bytes: int) -> int:
        template_da = self.level_templates[lvl][name]
        region = self._region_shape(lvl, name, max_region_bytes)
        return math.prod(math.ceil(n / r) for n, r in zip(template_da.shape, region))

    def write(
        self,
        store: Any,
        *,
        mode: str = "w",
        max_workers: int | None = None,
        levels: list[int] | None = None,
        max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
        progress: bool = False,
    ) -> None:
        """Compute and write pyramid levels to a Zarr store.

        Level 0 is streamed region by region from the source dataset; each
        subsequent level is block-reduced from the previously written level,
        streaming shard-sized regions through the Rust kernel on a thread
        pool. Levels are written sequentially (each reads the previous one);
        variables within a level are processed in parallel on a shared pool.
        For bounded memory on large stores, open the source lazily (e.g.
        ``xr.open_zarr(store, chunks=None)``).

        Args:
            store: Anything zarr-python accepts — a local path,
                ``ObjectStore``, or icechunk session store.
            mode: Zarr open mode for the root group. Use ``"a"`` when
                writing a subset of levels so the root group and any
                pre-existing levels are preserved.
            max_workers: Thread pool size for region processing. ``None``
                derives a default from the CPU count and available memory
                (peak memory is roughly ``max_workers * 5 * region_bytes``).
            levels: Subset of levels to write (e.g. ``[1, 2]``).
                Defaults to all levels.
            max_region_bytes: Memory budget per level-0 copy region. Regions
                are widened to cover whole source chunks when that fits the
                budget, so each source chunk is read once.
            progress: Show a tqdm progress bar over written regions
                (requires ``tqdm``).

        Examples:
            Write all levels to a local store:

            ```python
            pyramid.write("pyramid.zarr")
            ```

            Rewrite the coarsened levels, preserving level 0:

            ```python
            pyramid.write("pyramid.zarr", mode="a", levels=[1, 2])
            ```
        """
        if levels is not None:
            invalid = sorted(set(levels) - set(self.level_templates))
            if invalid:
                raise ValueError(
                    f"invalid levels {invalid}; pyramid has levels 0-{self.levels - 1}"
                )

        write_levels = list(range(self.levels)) if levels is None else list(levels)
        spatial_vars = self._spatial_vars()

        pbar = None
        on_region: Callable[[], None] | None = None
        if progress:
            total = sum(
                self._region_count(lvl, name, max_region_bytes)
                for lvl in write_levels
                for name in spatial_vars
            )
            pbar = _progress_bar(total)
            on_region = pbar.update

        root = zarr.open_group(store, mode=mode, zarr_format=3)
        root.attrs.update(self.attrs)

        try:
            for lvl in write_levels:
                template = self.level_templates[lvl]
                # coords + non-spatial vars + level attrs via xarray
                template.drop_vars(spatial_vars, errors="ignore").to_zarr(
                    store, group=str(lvl), mode="a", zarr_format=3, consolidated=False
                )
                if not spatial_vars:
                    continue
                level_group = root[str(lvl)]

                workers = max_workers
                if workers is None:
                    workers = default_max_workers(
                        max(
                            self._region_bytes(lvl, name, max_region_bytes)
                            for name in spatial_vars
                        )
                    )
                with ThreadPoolExecutor(workers) as ex:
                    futures = [
                        future
                        for name in spatial_vars
                        for future in self._write_var(
                            root,
                            level_group,
                            lvl,
                            name,
                            max_region_bytes,
                            executor=ex,
                            on_region=on_region,
                        )
                    ]
                    for future in futures:
                        future.result()
        finally:
            if pbar is not None:
                pbar.close()

    def _write_var(
        self,
        root: zarr.Group,
        level_group: zarr.Group,
        lvl: int,
        name: str,
        max_region_bytes: int,
        *,
        executor: ThreadPoolExecutor,
        on_region: Callable[[], None] | None,
    ) -> list[Future[None]]:
        template_da = self.level_templates[lvl][name]
        source_da = self.source[name]
        fill = _to_python(self.fill_values.get(name))

        attrs = _to_python(dict(template_da.attrs))
        extra_coords = [c for c in source_da.coords if c not in source_da.dims]
        if extra_coords:
            attrs["coordinates"] = " ".join(extra_coords)

        enc = self.encoding[f"/{lvl}"][name]
        dst = level_group.create_array(
            name=name,
            shape=template_da.shape,
            dtype=template_da.dtype,
            chunks=enc["chunks"],
            shards=enc.get("shards"),
            dimension_names=template_da.dims,
            attributes=attrs,
            fill_value=fill,
            overwrite=True,
        )

        if lvl == 0:
            return copy_array(
                source_da.variable,
                dst,
                source_chunks=source_chunks(source_da),
                max_region_bytes=max_region_bytes,
                executor=executor,
                on_region=on_region,
            )
        stride = tuple(
            2 if d in (self.x_dim, self.y_dim) else 1 for d in template_da.dims
        )
        return downsample_level(
            root[f"{lvl - 1}/{name}"],
            dst,
            stride=stride,
            method=self.method,
            fill_value=fill,
            executor=executor,
            on_region=on_region,
        )
