"""Python bridge to VectorCraft's headless CLI and live ``--control`` protocol."""

from .client import VectorCraftClient
from .discovery import LiveInstance, connect_first, discover_instances
from .errors import CliNotFoundError, ControlError, NoInstanceError, VectorCraftError
from .figures import export_plotly_figure, write_plotly_svg
from .find import ObjectQuery, bounds_xywh
from .layers import Layers, layer_geometry, replace_layers
from .live import LiveVectorCraft
from .measure import intrinsic_size_pts
from .place import PlaceSpec, place_into_template
from .record import Actions, CommandRecorder, load_recipe, play_action, record_action, replay_recipe, save_recipe
from .text import TextHelpers, find_text_objs_grouped_by_font_size, set_text_font_size_centered

__all__ = [
    "Actions",
    "CliNotFoundError",
    "CommandRecorder",
    "ControlError",
    "LiveInstance",
    "Layers",
    "LiveVectorCraft",
    "NoInstanceError",
    "ObjectQuery",
    "PlaceSpec",
    "TextHelpers",
    "VectorCraftClient",
    "VectorCraftError",
    "bounds_xywh",
    "connect_first",
    "discover_instances",
    "export_plotly_figure",
    "find_text_objs_grouped_by_font_size",
    "intrinsic_size_pts",
    "layer_geometry",
    "load_recipe",
    "place_into_template",
    "play_action",
    "record_action",
    "replay_recipe",
    "replace_layers",
    "save_recipe",
    "set_text_font_size_centered",
    "write_plotly_svg",
]

__version__ = "0.1.0"
