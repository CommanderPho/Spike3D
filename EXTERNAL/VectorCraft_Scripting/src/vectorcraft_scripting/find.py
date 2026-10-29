"""Object search criteria and client-side bounds/size filters for live documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


@dataclass
class ObjectQuery:
    """Find nodes in the active document.

    Native ``document.find`` filters: ``name`` (substring), ``kind`` (exact panel label), ``text``, ``limit``.
    Client-side filters applied after enrichment: ``name_regex``, width/height from ``bounds``, ``visible``, ``locked``.
    """

    name: Optional[str] = None
    kind: Optional[str] = None
    text: Optional[str] = None
    limit: Optional[int] = None
    name_regex: Optional[str] = None
    min_width: Optional[float] = None
    max_width: Optional[float] = None
    min_height: Optional[float] = None
    max_height: Optional[float] = None
    visible: Optional[bool] = None
    locked: Optional[bool] = None


    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObjectQuery":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


    def has_native_find_filters(self) -> bool:
        return any(v is not None for v in (self.name, self.kind, self.text))


    def has_size_or_flag_filters(self) -> bool:
        return any(v is not None for v in (self.name_regex, self.min_width, self.max_width, self.min_height, self.max_height, self.visible, self.locked))


    def to_find_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if self.name is not None:
            params["name"] = self.name
        if self.kind is not None:
            params["kind"] = self.kind
        if self.text is not None:
            params["text"] = self.text
        if self.limit is not None:
            params["limit"] = self.limit
        return params


def coerce_query(query: Union[ObjectQuery, Mapping[str, Any]]) -> ObjectQuery:
    if isinstance(query, ObjectQuery):
        return query
    return ObjectQuery.from_mapping(query)


def bounds_xywh(bounds: Any) -> Optional[Dict[str, float]]:
    """Interpret node ``bounds`` as ``{x, y, width, height}``.

    Accepts the dict form live ``document.inspect`` returns and the ``[x, y, w, h]`` list form.
    """
    if bounds is None:
        return None
    if isinstance(bounds, dict):
        try:
            return dict(x=float(bounds['x']), y=float(bounds['y']), width=float(bounds['width']), height=float(bounds['height']))
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
        try:
            return dict(x=float(bounds[0]), y=float(bounds[1]), width=float(bounds[2]), height=float(bounds[3]))
        except (TypeError, ValueError):
            return None
    return None


def bounds_size(bounds: Any) -> Optional[tuple[float, float]]:
    """Interpret node ``bounds`` as ``(width, height)`` from dict or ``[x, y, w, h]``."""
    xywh = bounds_xywh(bounds)
    if xywh is None:
        return None
    return (xywh['width'], xywh['height'])


def node_matches_query(node: Mapping[str, Any], query: ObjectQuery) -> bool:
    if query.name_regex is not None:
        name = str(node.get("name") or "")
        if re.search(query.name_regex, name) is None:
            return False
    if query.visible is not None and bool(node.get("visible", True)) != query.visible:
        return False
    if query.locked is not None and bool(node.get("locked", False)) != query.locked:
        return False
    size = bounds_size(node.get("bounds"))
    if any(v is not None for v in (query.min_width, query.max_width, query.min_height, query.max_height)):
        if size is None:
            return False
        width, height = size
        if query.min_width is not None and width < query.min_width:
            return False
        if query.max_width is not None and width > query.max_width:
            return False
        if query.min_height is not None and height < query.min_height:
            return False
        if query.max_height is not None and height > query.max_height:
            return False
    return True


def filter_nodes(nodes: Sequence[Mapping[str, Any]], query: ObjectQuery) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for node in nodes:
        if node_matches_query(node, query):
            matched.append(dict(node))
    ## END for node in nodes...

    if query.limit is not None and query.limit >= 0:
        return matched[: query.limit]
    return matched
