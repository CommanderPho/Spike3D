"""Optional Plotly → SVG → VectorCraft place/export convenience helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from .client import VectorCraftClient
from .place import PlaceSpec, place_into_template


def write_plotly_svg(fig: Any, path: Union[str, Path], *, width: Optional[int] = None, height: Optional[int] = None) -> Path:
    """Write a Plotly figure to SVG via Kaleido. Requires optional deps: ``plotly``, ``kaleido``."""
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_kwargs: dict = {"format": "svg"}
    if width is not None:
        write_kwargs["width"] = width
    if height is not None:
        write_kwargs["height"] = height
    fig.write_image(str(out), **write_kwargs)
    return out


def export_plotly_figure(fig: Any, chart_svg: Union[str, Path], template: Union[str, Path], rect: Sequence[float], exports: Sequence[Union[str, Path]], *, client: Optional[VectorCraftClient] = None, crop: str = "crop", link: bool = False, scale: float = 1.0, width: Optional[int] = None, height: Optional[int] = None) -> List[Any]:
    """README workflow: write Plotly figure as SVG, place into ``template`` at ``rect``, export.

    ``rect`` is ``[x, y, width, height]`` in document points.
    """
    svg_path = write_plotly_svg(fig, chart_svg, width=width, height=height)
    place = PlaceSpec(path=svg_path, rect=rect, crop=crop, link=link)
    return place_into_template(template, [place], exports, client=client, scale=scale)
