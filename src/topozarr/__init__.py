from .coarsen import create_pyramid
from .metadata import ZarrLayerVarConfig
from .pyramid import CoarseningMethod, Pyramid
from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("topozarr")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["create_pyramid", "Pyramid", "CoarseningMethod", "ZarrLayerVarConfig"]
