"""Torch-RecHub: A PyTorch Toolbox for Recommendation Models."""

from importlib.metadata import PackageNotFoundError, metadata

try:
    _meta = metadata("torch-rechub")
except PackageNotFoundError:
    _meta = {}

__version__ = _meta.get("Version", "0.8.0")
__author__ = _meta.get("Author") or _meta.get("Author-email", "Datawhale")
__license__ = _meta.get("License") or _meta.get("License-Expression", "MIT")
__url__ = _meta.get("Home-page") or _meta.get("Project-URL", "").split(", ")[-1] or "https://github.com/datawhalechina/torch-rechub"

# Import public modules.
from . import basic, models, trainers, utils

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__url__",
    "basic",
    "models",
    "trainers",
    "utils",
]
