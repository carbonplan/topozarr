from .coarsen import CoarseningMethod, create_pyramid
from .metadata import ZarrLayerVarConfig
from .pyramid import Pyramid
from .write import write_pyramid
from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("topozarr")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["create_pyramid", "write_pyramid", "Pyramid", "CoarseningMethod", "ZarrLayerVarConfig"]
