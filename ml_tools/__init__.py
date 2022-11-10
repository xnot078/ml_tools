__verson__ = "1.0.0"

__all__ = [
    "version",
    "chain",
    "save_estimator",
    "open_estimator"
]

from .base import chain
from skelarn2json import save_estimator
from skelarn2json import open_estimator
