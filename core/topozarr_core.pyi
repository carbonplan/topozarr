from typing import Any

import numpy.typing as npt

def block_reduce(
    a: npt.NDArray[Any],
    stride: list[int] | tuple[int, ...],
    method: str,
    fill_value: float | None = None,
    skipna: bool = True,
) -> npt.NDArray[Any]: ...

class RustWriter:
    def __init__(self, url: str, options: dict[str, str] | None = None) -> None: ...
    def write_region(
        self, path: str, start: list[int], block: npt.NDArray[Any]
    ) -> None: ...
    def stats(self) -> dict[str, float | int]: ...
    def flush(self) -> None: ...
