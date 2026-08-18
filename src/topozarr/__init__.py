from importlib.metadata import PackageNotFoundError, version

from .coarsen import create_pyramid
from .geozarr import attach_geozarr_metadata
from .metadata import ZarrLayerVarConfig, recommend_encoding
from .pyramid import CoarseningMethod, Pyramid

try:
    __version__ = version("topozarr")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "create_pyramid",
    "attach_geozarr_metadata",
    "recommend_encoding",
    "Pyramid",
    "CoarseningMethod",
    "ZarrLayerVarConfig",
]
