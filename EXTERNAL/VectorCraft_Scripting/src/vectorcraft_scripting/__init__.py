"""Python bridge to VectorCraft's headless CLI and live ``--control`` protocol."""

from .client import VectorCraftClient
from .discovery import LiveInstance, connect_first, discover_instances
from .errors import CliNotFoundError, ControlError, NoInstanceError, VectorCraftError
from .figures import export_plotly_figure, write_plotly_svg
from .find import ObjectQuery
from .live import LiveVectorCraft
from .place import PlaceSpec, place_into_template

__all__ = [
    "CliNotFoundError",
    "ControlError",
    "LiveInstance",
    "LiveVectorCraft",
    "NoInstanceError",
    "ObjectQuery",
    "PlaceSpec",
    "VectorCraftClient",
    "VectorCraftError",
    "connect_first",
    "discover_instances",
    "export_plotly_figure",
    "place_into_template",
    "write_plotly_svg",
]

__version__ = "0.1.0"
