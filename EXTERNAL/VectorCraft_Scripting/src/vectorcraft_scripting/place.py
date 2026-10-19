"""Helpers for ``file.place`` into VectorCraft templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .cli import abs_path
from .client import CommandSpec, VectorCraftClient


@dataclass
class PlaceSpec:
    """Parameters for a single ``file.place`` command.

    ``rect`` is ``[x, y, width, height]`` in document points (from the artboard top-left).
    ``at`` is ``[x, y]`` centre placement when ``rect`` is omitted.
    """

    path: Union[str, Path]
    rect: Optional[Sequence[float]] = None
    at: Optional[Sequence[float]] = None
    crop: Optional[str] = "crop"
    link: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"path": abs_path(self.path), "link": self.link}
        if self.rect is not None:
            params["rect"] = list(self.rect)
        if self.at is not None:
            params["at"] = list(self.at)
        if self.crop is not None:
            params["crop"] = self.crop
        params.update(self.extra)
        return params


    def as_command(self) -> Tuple[str, Dict[str, Any]]:
        return ("file.place", self.to_params())


def _coerce_place(place: Union[PlaceSpec, Mapping[str, Any]]) -> PlaceSpec:
    if isinstance(place, PlaceSpec):
        return place
    data = dict(place)
    path = data.pop("path")
    rect = data.pop("rect", None)
    at = data.pop("at", None)
    crop = data.pop("crop", "crop")
    link = data.pop("link", False)
    return PlaceSpec(path=path, rect=rect, at=at, crop=crop, link=link, extra=data)


def place_into_template(template: Union[str, Path], places: Sequence[Union[PlaceSpec, Mapping[str, Any]]], exports: Sequence[Union[str, Path]], *, client: Optional[VectorCraftClient] = None, scale: float = 1.0) -> List[Any]:
    """Open ``template``, place each file, then write each path in ``exports``."""
    vc = client or VectorCraftClient()
    commands: List[CommandSpec] = [_coerce_place(place).as_command() for place in places]
    return vc.run(input=template, commands=commands, exports=exports, scale=scale)
