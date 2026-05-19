from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from xarray import DataTree


@dataclass
class Pyramid:
    datatree: DataTree
    encoding: dict[str, Any]
    x_dim: str = "x"
    y_dim: str = "y"
    method: str = "mean"

    @property
    def dt(self) -> DataTree:
        return self.datatree
