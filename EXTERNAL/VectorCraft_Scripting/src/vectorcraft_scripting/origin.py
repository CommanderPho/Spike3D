"""Capture a registration (origin) point from the Direct Selection and align/replace by that point.

Typical interactive flow matching a selected path point on the object being replaced and the
corresponding point on the replacement:

1. Direct-select the destination anchor on the existing object → ``capture_origin_from_selection``
2. Direct-select the matching anchor on the replacement → ``capture_origin_from_selection``
3. ``replace_by_origin(live, target, source)`` moves the source so those points coincide and
   optionally hides the target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .errors import ControlError
from .find import bounds_xywh
from .live import LiveVectorCraft
from .measure import intrinsic_size_pts


PROBE_DX_PT: float = 7.777


@dataclass
class OriginPoint:
    """A registration point in document coordinates (y down, artboard top-left origin)."""

    x: float
    y: float
    object_ids: List[Any] = field(default_factory=list)
    anchor_refs: Dict[Any, List[Tuple[int, int]]] = field(default_factory=dict)  # id -> [(subpath, anchor), ...]
    document_title: Optional[str] = None
    bounds: Optional[Dict[str, float]] = None
    fully_selected: bool = False


    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['anchor_refs'] = {str(k): [list(ref) for ref in refs] for k, refs in self.anchor_refs.items()}
        return data


    @classmethod
    def from_xy(cls, x: float, y: float, *, document_title: Optional[str] = None) -> "OriginPoint":
        return cls(x=float(x), y=float(y), document_title=document_title)


class Origin:
    """Helpers for origin-point capture, file placement, and origin-aligned replace."""


    @classmethod
    def _subfn_extract_path_data(cls, node: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the ``{subpaths: [...]}`` payload from a ``document.node`` reply."""
        kind = node.get('kind')
        if isinstance(kind, dict):
            if isinstance(kind.get('path'), dict) and ('subpaths' in kind['path']):
                return kind['path']
            if 'subpaths' in kind:
                return dict(kind)
        path = node.get('path')
        if isinstance(path, dict) and ('subpaths' in path):
            return path
        return None


    @classmethod
    def _subfn_iter_descendant_nodes(cls, node: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Flatten ``document.node`` children (groups / compounds) depth-first, including ``node`` itself."""
        out: List[Dict[str, Any]] = [dict(node)]
        kind = node.get('kind')
        children: List[Any] = []
        if isinstance(kind, dict):
            children = list(kind.get('children') or [])
        if not children:
            children = list(node.get('children') or node.get('layers') or [])
        for child in children:
            if isinstance(child, dict):
                out.extend(cls._subfn_iter_descendant_nodes(child))
        ## END for child in children...
        return out


    @classmethod
    def _subfn_path_ids_under(cls, live: LiveVectorCraft, node_id: Any) -> List[Any]:
        """Path object ids at or under ``node_id`` (Direct-Selected anchors often live on a group child)."""
        node = live.get_node(node_id, summary=False)
        assert isinstance(node, dict), f"document.node did not return a dict for id={node_id!r}: {node!r}"
        path_ids: List[Any] = []
        for descendant in cls._subfn_iter_descendant_nodes(node):
            if cls._subfn_extract_path_data(descendant) is None:
                continue
            did = descendant.get('id')
            if did is not None:
                path_ids.append(did)
        ## END for descendant in cls._subfn_iter_descendant_nodes(node)...
        return path_ids


    @classmethod
    def _subfn_anchor_positions(cls, live: LiveVectorCraft, node_id: Any) -> Dict[Tuple[int, int], Tuple[float, float]]:
        node = live.get_node(node_id, summary=False)
        assert isinstance(node, dict), f"document.node did not return a dict for id={node_id!r}: {node!r}"
        path = cls._subfn_extract_path_data(node)
        if path is None:
            return {}
        out: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for si, sp in enumerate(path.get('subpaths') or []):
            if not isinstance(sp, dict):
                continue
            for ai, anchor in enumerate(sp.get('anchors') or []):
                if not isinstance(anchor, dict):
                    continue
                p = anchor.get('p')
                if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                    continue
                out[(int(si), int(ai))] = (float(p[0]), float(p[1]))
            ## END for ai, anchor in enumerate(sp.get('anchors') or [])...
        ## END for si, sp in enumerate(path.get('subpaths') or [])...
        return out


    @classmethod
    def _subfn_detect_selected_anchors(cls, live: LiveVectorCraft, path_ids: Sequence[Any]) -> Tuple[Dict[Any, List[Tuple[int, int]]], List[Tuple[float, float]], bool]:
        """Probe ``path.moveAnchors`` once across ``path_ids``; return refs, points, and fully_selected.

        VectorCraft's ``document.inspect`` lists selected object ids but not which Direct-Selected anchors
        are active. Moving by a known delta and diffing anchor positions recovers that set; undo restores.
        """
        before_by_id: Dict[Any, Dict[Tuple[int, int], Tuple[float, float]]] = {}
        for path_id in path_ids:
            before_by_id[path_id] = cls._subfn_anchor_positions(live, path_id)
        ## END for path_id in path_ids...
        if not any(before_by_id.values()):
            return {}, [], False

        live.execute('path.moveAnchors', {'dx': PROBE_DX_PT, 'dy': 0.0})
        try:
            after_by_id: Dict[Any, Dict[Tuple[int, int], Tuple[float, float]]] = {path_id: cls._subfn_anchor_positions(live, path_id) for path_id in path_ids}
        finally:
            live.execute('edit.undo', {})

        anchor_refs: Dict[Any, List[Tuple[int, int]]] = {}
        points: List[Tuple[float, float]] = []
        total_anchors = 0
        for path_id, before in before_by_id.items():
            total_anchors += len(before)
            after = after_by_id.get(path_id) or {}
            moved: List[Tuple[int, int]] = []
            for ref, (x0, y0) in before.items():
                x1, y1 = after.get(ref, (x0, y0))
                if (abs(x1 - x0 - PROBE_DX_PT) < 1e-4 and abs(y1 - y0) < 1e-4) or (abs(x1 - x0) > 1e-4 or abs(y1 - y0) > 1e-4):
                    moved.append(ref)
                    points.append((x0, y0))
            ## END for ref, (x0, y0) in before.items()...
            if moved:
                anchor_refs[path_id] = sorted(moved)
        ## END for path_id, before in before_by_id.items()...

        if not points:
            return {}, [], False
        fully_selected = (sum(len(refs) for refs in anchor_refs.values()) == total_anchors) and (total_anchors > 0)
        return anchor_refs, points, fully_selected


    @classmethod
    def capture_from_selection(cls, live: LiveVectorCraft) -> OriginPoint:
        """Read the current Direct Selection as an ``OriginPoint`` (centroid of selected anchors).

        Requires at least one selected object. Prefer a single Direct-Selected path anchor (as in the
        replace-registration screenshots); a fully selected path uses the centroid of all of its anchors.
        Selected groups are walked so anchors on child paths are found.
        """
        info = live.active_document_info()
        title: Optional[str] = str(info.get('title') or '') if isinstance(info, dict) else None
        tree = live.inspect_document(depth=0, childLimit=0)
        assert isinstance(tree, dict), f"document.inspect did not return a dict: {tree!r}"
        selection: List[Any] = list(tree.get('selection') or [])
        assert selection, "no selection — Direct-Select the origin path point first"

        path_ids: List[Any] = []
        seen = set()
        primary_bounds: Optional[Dict[str, float]] = None
        for node_id in selection:
            for path_id in cls._subfn_path_ids_under(live, node_id):
                if path_id in seen:
                    continue
                seen.add(path_id)
                path_ids.append(path_id)
            ## END for path_id in cls._subfn_path_ids_under(live, node_id)...
            if primary_bounds is None:
                try:
                    primary_bounds = live.node_bounds(node_id)
                except (AssertionError, ControlError, KeyError, TypeError):
                    primary_bounds = bounds_xywh((live.get_node(node_id, summary=True, depth=0) or {}).get('bounds'))
        ## END for node_id in selection...

        anchor_refs, points, fully_selected = cls._subfn_detect_selected_anchors(live, path_ids) if path_ids else ({}, [], False)

        if points:
            x = sum(p[0] for p in points) / float(len(points))
            y = sum(p[1] for p in points) / float(len(points))
            return OriginPoint(x=x, y=y, object_ids=list(selection), anchor_refs=anchor_refs, document_title=title, bounds=primary_bounds, fully_selected=fully_selected)

        # Non-path selection: use selectionBounds, else first object bounds top-left.
        sel_bounds = bounds_xywh(tree.get('selectionBounds')) or primary_bounds
        assert sel_bounds is not None, f"selection {selection!r} has no path anchors and no bounds to use as origin"
        return OriginPoint(x=float(sel_bounds['x']), y=float(sel_bounds['y']), object_ids=list(selection), anchor_refs={}, document_title=title, bounds=sel_bounds, fully_selected=True)


    @classmethod
    def place_rect_for_origin(cls, *, target_xy: Sequence[float], source_origin_xy: Sequence[float], source_size: Sequence[float], scale: float = 1.0) -> Dict[str, float]:
        """Build a ``file.place`` rect so ``source_origin_xy`` (in unscaled source space) lands on ``target_xy``."""
        tx, ty = float(target_xy[0]), float(target_xy[1])
        ox, oy = float(source_origin_xy[0]), float(source_origin_xy[1])
        sw, sh = float(source_size[0]), float(source_size[1])
        assert scale > 0.0, f"scale must be positive, got {scale!r}"
        assert sw > 0.0 and sh > 0.0, f"source_size must be positive, got {source_size!r}"
        return {'x': tx - ox * scale, 'y': ty - oy * scale, 'width': sw * scale, 'height': sh * scale}


    @classmethod
    def place_file_at_origin(cls, live: LiveVectorCraft, path: Union[str, Path], target_xy: Sequence[float], source_origin_xy: Sequence[float], *, scale: float = 1.0, source_size: Optional[Sequence[float]] = None, name: Optional[str] = None, crop: Optional[str] = None, link: bool = False) -> Tuple[List[Any], Dict[str, float], Dict[str, float]]:
        """``file.place`` so the source registration point lands on ``target_xy``.

        Returns ``(placed_ids, place_rect, actual_bounds)``.
        """
        file_path: Path = Path(path)
        size = tuple(source_size) if source_size is not None else intrinsic_size_pts(file_path)
        assert size is not None, f"could not read intrinsic size of '{file_path.as_posix()}' — pass source_size=(w, h) explicitly"
        place_rect = cls.place_rect_for_origin(target_xy=target_xy, source_origin_xy=source_origin_xy, source_size=size, scale=scale)
        if crop is None and (file_path.suffix.lower() in ('.svg', '.svgz')):
            crop = None
        elif crop is None:
            crop = 'crop'
        placed_ids, actual = live.place_file(file_path, place_rect, name=name, crop=crop, link=link)
        return placed_ids, place_rect, actual


    @classmethod
    def _subfn_ensure_document(cls, live: LiveVectorCraft, title: Optional[str]) -> None:
        if title:
            live.select_document(title)


    @classmethod
    def _subfn_bring_source_into_target_doc(cls, live: LiveVectorCraft, target: OriginPoint, source: OriginPoint) -> Tuple[List[Any], Tuple[float, float]]:
        """If ``source`` lives in another tab, copy-paste it into the target document; return (ids, origin_xy)."""
        same_doc = (not target.document_title) or (not source.document_title) or (target.document_title == source.document_title)
        if same_doc:
            cls._subfn_ensure_document(live, target.document_title or source.document_title)
            return list(source.object_ids), source.xy

        assert source.object_ids, "cross-document replace_by_origin needs source.object_ids (capture from a selection)"
        cls._subfn_ensure_document(live, source.document_title)
        live.execute('select.set', {'ids': list(source.object_ids)})
        live.execute('edit.copy', {})
        cls._subfn_ensure_document(live, target.document_title)
        live.execute('edit.pasteInPlace', {})
        tree = live.inspect_document(depth=0, childLimit=0)
        pasted_ids: List[Any] = list((tree or {}).get('selection') or []) if isinstance(tree, dict) else []
        assert pasted_ids, f"edit.pasteInPlace produced no selection after copying from {source.document_title!r}"
        # pasteInPlace keeps page coordinates; origin xy is unchanged across docs that share artboard space.
        return pasted_ids, source.xy


    @classmethod
    def align_by_origin(cls, live: LiveVectorCraft, target: OriginPoint, source: OriginPoint, *, move_ids: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        """Move ``source`` (or ``move_ids``) so ``source.xy`` lands on ``target.xy``. Returns move diagnostics."""
        ids, source_xy = cls._subfn_bring_source_into_target_doc(live, target, source)
        if move_ids is not None:
            ids = list(move_ids)
        assert ids, "align_by_origin needs source object ids (or move_ids)"
        cls._subfn_ensure_document(live, target.document_title)
        dx: float = float(target.x) - float(source_xy[0])
        dy: float = float(target.y) - float(source_xy[1])
        if (abs(dx) > 1e-9) or (abs(dy) > 1e-9):
            live.execute('object.move', {'ids': ids, 'dx': dx, 'dy': dy})
        return {'moved_ids': ids, 'dx': dx, 'dy': dy, 'target_xy': target.xy, 'source_xy_before': source_xy}


    @classmethod
    def replace_by_origin(cls, live: LiveVectorCraft, target: OriginPoint, source: OriginPoint, *, hide_target: bool = True, select_result: bool = True) -> Dict[str, Any]:
        """Align ``source`` onto ``target``'s origin point and optionally hide the target object(s).

        Does not delete the target — hide only — so the replace stays one undo-friendly step when wrapped
        in ``live.undo_group(...)``.
        """
        align = cls.align_by_origin(live, target, source)
        if hide_target and target.object_ids:
            cls._subfn_ensure_document(live, target.document_title)
            live.execute('object.setProps', {'ids': list(target.object_ids), 'visible': False})
        if select_result and align['moved_ids']:
            live.execute('select.set', {'ids': list(align['moved_ids'])})
        return {'align': align, 'hidden_target_ids': (list(target.object_ids) if hide_target else []), 'target': target.to_dict(), 'source': source.to_dict()}


def capture_origin_from_selection(live: LiveVectorCraft) -> OriginPoint:
    """Read the current Direct Selection as an ``OriginPoint`` (centroid of selected anchors)."""
    return Origin.capture_from_selection(live)


def place_rect_for_origin(*, target_xy: Sequence[float], source_origin_xy: Sequence[float], source_size: Sequence[float], scale: float = 1.0) -> Dict[str, float]:
    """Build a ``file.place`` rect so ``source_origin_xy`` (in unscaled source space) lands on ``target_xy``."""
    return Origin.place_rect_for_origin(target_xy=target_xy, source_origin_xy=source_origin_xy, source_size=source_size, scale=scale)


def place_file_at_origin(live: LiveVectorCraft, path: Union[str, Path], target_xy: Sequence[float], source_origin_xy: Sequence[float], *, scale: float = 1.0, source_size: Optional[Sequence[float]] = None, name: Optional[str] = None, crop: Optional[str] = None, link: bool = False) -> Tuple[List[Any], Dict[str, float], Dict[str, float]]:
    """``file.place`` so the source registration point lands on ``target_xy``."""
    return Origin.place_file_at_origin(live, path, target_xy, source_origin_xy, scale=scale, source_size=source_size, name=name, crop=crop, link=link)


def align_by_origin(live: LiveVectorCraft, target: OriginPoint, source: OriginPoint, *, move_ids: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Move ``source`` so ``source.xy`` lands on ``target.xy``."""
    return Origin.align_by_origin(live, target, source, move_ids=move_ids)


def replace_by_origin(live: LiveVectorCraft, target: OriginPoint, source: OriginPoint, *, hide_target: bool = True, select_result: bool = True) -> Dict[str, Any]:
    """Align ``source`` onto ``target``'s origin point and optionally hide the target object(s)."""
    return Origin.replace_by_origin(live, target, source, hide_target=hide_target, select_result=select_result)
