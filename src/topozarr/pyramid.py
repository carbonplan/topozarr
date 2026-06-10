from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import xarray as xr
import zarr

from .engine import copy_array, downsample_level

CoarseningMethod = Literal["mean", "max", "min", "sum"]


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
    """Result of :func:`create_pyramid` — a write plan for a multiscale Zarr pyramid.

    Attributes:
        source: The original (level 0) dataset.
        level_templates: Per-level datasets carrying real coordinates and
            attrs; spatial data variables are zero-cost placeholders with the
            correct shape/dtype (their data is computed during :meth:`write`).
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

    def write(
        self,
        store: Any,
        *,
        mode: str = "w",
        max_workers: int | None = None,
        levels: list[int] | None = None,
    ) -> None:
        """Write the pyramid to a Zarr store.

        Level 0 is copied from the source dataset; each subsequent level is
        block-reduced from the previously written level, streaming
        shard-sized regions through the Rust kernel. ``store`` is anything
        zarr-python accepts (path, ``ObjectStore``, icechunk session store).

        ``levels`` restricts which levels are written; defaults to all. When
        writing a subset (e.g. ``levels=[1, 2]``) use ``mode="a"`` so the
        root group and any pre-existing levels are preserved.
        """
        root = zarr.open_group(store, mode=mode, zarr_format=3)
        root.attrs.update(self.attrs)
        spatial_vars = self._spatial_vars()

        for lvl in range(self.levels) if levels is None else levels:
            template = self.level_templates[lvl]
            # coords + non-spatial vars + level attrs via xarray
            template.drop_vars(spatial_vars, errors="ignore").to_zarr(
                store, group=str(lvl), mode="a", zarr_format=3, consolidated=False
            )
            level_group = root[str(lvl)]
            for name in spatial_vars:
                self._write_var(root, level_group, lvl, name, max_workers)

    def _write_var(
        self,
        root: zarr.Group,
        level_group: zarr.Group,
        lvl: int,
        name: str,
        max_workers: int | None,
    ) -> None:
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
            copy_array(np.asarray(source_da.values), dst, max_workers=max_workers)
        else:
            stride = tuple(
                2 if d in (self.x_dim, self.y_dim) else 1 for d in template_da.dims
            )
            downsample_level(
                root[f"{lvl - 1}/{name}"],
                dst,
                stride=stride,
                method=self.method,
                fill_value=fill,
                max_workers=max_workers,
            )
