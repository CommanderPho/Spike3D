"""Live helpers for VectorCraft type objects (font size with center-preserving layout)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple, Union

from .live import LiveVectorCraft


class TextHelpers:

    @classmethod
    def set_text_font_size_centered(cls, live: LiveVectorCraft, ids: Union[Any, Sequence[Any]], size_pt: float) -> List[Dict[str, Any]]:
        """Set type objects to ``size_pt`` via ``text.setStyle``, then ``object.move`` so each bounds center stays fixed.

        Callers should wrap with ``live.undo_group(...)`` for a single undo step. Returns one result dict per id.
        """
        assert float(size_pt) > 0, f"size_pt must be > 0, got {size_pt!r}"
        size_pt = float(size_pt)

        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            id_list: List[Any] = [ids]
        else:
            id_list = list(ids)
        assert id_list, "ids must not be empty"

        def _subfn_bounds_center(xywh: Dict[str, float]) -> Tuple[float, float]:
            return (xywh['x'] + xywh['width'] / 2.0, xywh['y'] + xywh['height'] / 2.0)

        results: List[Dict[str, Any]] = []
        for node_id in id_list:
            node = live.get_node(node_id, summary=True, depth=0)
            assert isinstance(node, dict), f"document.node did not return a dict for id={node_id!r}: {node!r}"
            kind = str(node.get('kind') or '')
            assert kind.lower() == 'type', f"id={node_id!r} kind={kind!r} is not Type — set_text_font_size_centered only accepts type objects"

            bounds_before = live.node_bounds(node_id)
            cx, cy = _subfn_bounds_center(bounds_before)

            reply = live.execute('text.setStyle', {'ids': [node_id], 'size': size_pt})

            bounds_after_style = live.node_bounds(node_id)
            after_cx, after_cy = _subfn_bounds_center(bounds_after_style)
            dx = cx - after_cx
            dy = cy - after_cy
            if dx != 0.0 or dy != 0.0:
                live.execute('object.move', {'ids': [node_id], 'dx': dx, 'dy': dy})

            bounds_after = live.node_bounds(node_id)
            row: Dict[str, Any] = {'id': node_id, 'size_pt': size_pt, 'dx': dx, 'dy': dy, 'bounds_before': bounds_before, 'bounds_after': bounds_after}
            if isinstance(reply, dict):
                if 'overflow' in reply:
                    row['overflow'] = reply['overflow']
                if 'warning' in reply:
                    row['warning'] = reply['warning']
            results.append(row)
        ## END for node_id in id_list...

        return results


    @classmethod
    def find_text_objs_grouped_by_font_size(cls, live: LiveVectorCraft, *, limit: int = 10000, effective: bool = True) -> Dict[str, Any]:
        """Find all Type objects and group them by effective (or stored) run font sizes.

        Returns ``{'by_size': {size_pt: [rows…]}, 'mixed': [rows…], 'total': N}``. Each row is
        ``{id, name, text, sizes, runs}``. Multi-size objects appear in ``mixed`` and in every size bucket.
        """
        found = live.execute('document.find', {'kind': 'Type', 'limit': int(limit)})
        matches: List[Dict[str, Any]] = list((found or {}).get('matches') or []) if isinstance(found, dict) else list(found or [])

        by_size: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        mixed_font_size: List[Dict[str, Any]] = []

        for m in matches:
            if not isinstance(m, dict) or 'id' not in m:
                continue
            nid = m['id']
            rng = live.execute('text.getRange', {'id': nid, 'effective': bool(effective)})
            runs = list((rng or {}).get('runs') or []) if isinstance(rng, dict) else []
            sizes = sorted({
                float(r['style']['size'])
                for r in runs
                if isinstance(r, dict) and isinstance(r.get('style'), dict) and r['style'].get('size') is not None
            })
            row: Dict[str, Any] = {'id': nid, 'name': m.get('name'), 'text': (rng.get('text') if isinstance(rng, dict) else None), 'sizes': sizes, 'runs': len(runs)}
            if len(sizes) == 1:
                by_size[sizes[0]].append(row)
            elif sizes:
                mixed_font_size.append(row)
                for s in sizes:
                    by_size[s].append(row)
                ## END for s in sizes...
        ## END for m in matches...

        return {'by_size': dict(by_size), 'mixed_font_size': mixed_font_size, 'total': len(matches)}


def set_text_font_size_centered(live: LiveVectorCraft, ids: Union[Any, Sequence[Any]], size_pt: float) -> List[Dict[str, Any]]:
    """Set type objects to ``size_pt`` via ``text.setStyle``, then ``object.move`` so each bounds center stays fixed."""
    return TextHelpers.set_text_font_size_centered(live, ids, size_pt)


def find_text_objs_grouped_by_font_size(live: LiveVectorCraft, *, limit: int = 10000, effective: bool = True) -> Dict[str, Any]:
    """Find all Type objects and group them by effective (or stored) run font sizes."""
    return TextHelpers.find_text_objs_grouped_by_font_size(live, limit=limit, effective=effective)
