"""Python bridge to VectorCraft's headless ``vectorcraft-cli``."""

from .client import VectorCraftClient
from .errors import CliNotFoundError, VectorCraftError
from .figures import export_plotly_figure, write_plotly_svg
from .place import PlaceSpec, place_into_template

__all__ = [
    "CliNotFoundError",
    "PlaceSpec",
    "VectorCraftClient",
    "VectorCraftError",
    "export_plotly_figure",
    "place_into_template",
    "write_plotly_svg",
]

__version__ = "0.1.0"
