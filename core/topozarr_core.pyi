from typing import Any

import numpy.typing as npt

def block_reduce(
    a: npt.NDArray[Any],
    stride: list[int] | tuple[int, ...],
    method: str,
    fill_value: float | None = None,
    skipna: bool = True,
) -> npt.NDArray[Any]: ...
