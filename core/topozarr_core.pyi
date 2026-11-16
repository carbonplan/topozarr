from typing import Any

import numpy.typing as npt

def block_reduce(
    a: npt.NDArray[Any],
    stride: list[int] | tuple[int, ...],
    method: str,
    fill_value: float | None = None,
    skipna: bool = True,
) -> npt.NDArray[Any]: ...
def pack(
    x: npt.NDArray[Any],
    scale: float,
    offset: float,
    fill: int,
    dtype: str,
) -> npt.NDArray[Any]:
    """CF-pack a float32/float64 array to the integer dtype named by `dtype`."""

METHODS: tuple[str, ...]
"""Coarsening methods this kernel implements, in `block_reduce`'s `method` order."""
