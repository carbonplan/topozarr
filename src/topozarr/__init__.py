from .coarsen import create_pyramid
from .geozarr import attach_geozarr_metadata
from .metadata import ZarrLayerVarConfig
from .pyramid import CoarseningMethod, Pyramid
from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("topozarr")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "create_pyramid",
    "attach_geozarr_metadata",
    "Pyramid",
    "CoarseningMethod",
    "ZarrLayerVarConfig",
]
