from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from xarray import DataTree


@dataclass
class Pyramid:
    """Result of :func:`create_pyramid` — a multiscale DataTree paired with its encoding.

    Attributes:
        datatree: DataTree where each child node (``"0"``, ``"1"``, ...) holds one
            resolution level.  Level ``0`` is the original (highest) resolution;
            each subsequent level is coarsened by 2× per spatial dimension.
        encoding: Nested dict ``{path: {var: {"chunks": ..., "shards": ...}}}``
            suitable for passing directly as the ``encoding`` argument to
            ``DataTree.to_zarr``.
    """

    datatree: DataTree
    encoding: dict[str, Any]

    @property
    def dt(self) -> DataTree:
        return self.datatree
