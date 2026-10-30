"""Composite live-document workflows: layer geometry lookup and in-place figure replacement."""

from __future__ import annotations

import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .find import ObjectQuery, bounds_xywh
from .live import LiveVectorCraft
from .measure import intrinsic_size_pts

import pandas as pd


class Layers:

    @classmethod
    def _subfn_flatten_layer_nodes(cls, nodes: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            out.append(node)
            children = node.get('children') or node.get('layers') or []
            if isinstance(children, list) and children:
                out.extend(cls._subfn_flatten_layer_nodes(children))
        ## END for node in nodes...
        return out


    @classmethod
    def _subfn_parent_map(cls, nodes: List[Any], parent: Optional[Dict[str, Any]] = None) -> Dict[Any, Dict[str, Any]]:
        """Map each node id to ``{id, name, kind}`` of its Layers-panel parent (empty dict at the top level)."""
        index: Dict[Any, Dict[str, Any]] = {}
        parent_info: Dict[str, Any] = {} if parent is None else {'id': parent.get('id'), 'name': parent.get('name'), 'kind': parent.get('kind')}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get('id')
            if node_id is not None:
                index[node_id] = parent_info
            children = node.get('children') or node.get('layers') or []
            if isinstance(children, list) and children:
                index.update(cls._subfn_parent_map(children, parent=node))
        ## END for node in nodes...
        return index


    @classmethod
    def _subfn_kind_is_layer(cls, kind: Any) -> bool:
        return str(kind or '') in ('Layer', 'Sublayer')


    @classmethod
    def _subfn_node_geom_row(cls, query_name: str, node: Dict[str, Any]) -> Dict[str, Any]:
        xywh = bounds_xywh(node.get('bounds')) or {}
        return {'query': query_name, 'id': node.get('id'), 'name': node.get('name'), 'kind': node.get('kind'), 'x': xywh.get('x'), 'y': xywh.get('y'), 'width': xywh.get('width'), 'height': xywh.get('height'), 'visible': node.get('visible'), 'locked': node.get('locked')}


    @classmethod
    def layer_geometry(cls, live: LiveVectorCraft, layer_names: Sequence[str]) -> Any:
        """Lookup named layers in the active document; return a DataFrame indexed by ``name`` with x/y/width/height."""
        info = live.active_document_info()
        print(f"active document: {info.get('title')!r}  units={info.get('units')!r}")

        tree = live.inspect_document(depth=8, childLimit=500)
        flat_nodes = cls._subfn_flatten_layer_nodes(list(tree.get('layers') or []))

        rows: List[Dict[str, Any]] = []
        for name in layer_names:
            matches = live.find_objects(ObjectQuery(name=name, limit=50))
            match_ids = [m['id'] for m in matches if isinstance(m, dict) and 'id' in m]
            found: List[Dict[str, Any]] = []
            if match_ids:
                inspected = live.inspect_document(ids=match_ids, depth=0)
                found = list(inspected.get('nodes') or []) if isinstance(inspected, dict) else []
                if not found:
                    found = matches
            if not found:
                found = [n for n in flat_nodes if name in str(n.get('name') or '')]
            if not found:
                rows.append(cls._subfn_node_geom_row(name, {}))
                continue
            seen_ids = set()
            for node in found:
                node_id = node.get('id')
                if node_id in seen_ids:
                    continue
                if node_id is not None:
                    seen_ids.add(node_id)
                if (node.get('bounds') is None) and (node_id is not None):
                    node = live.get_node(node_id, summary=True, depth=0) or node
                rows.append(cls._subfn_node_geom_row(name, node))
            ## END for node in found...
        ## END for name in layer_names...

        layer_geom_df = pd.DataFrame(rows)
        layer_geom_df.set_index('name', drop=True, inplace=True)
        return layer_geom_df


    @classmethod
    def replace_layers(cls, live: LiveVectorCraft, layer_rects: Mapping[str, Mapping[str, float]], source_folder: Union[str, Path], *, expect_title: Optional[str] = None, backup_layers: bool = True, backup_suffix: Optional[str] = None, new_suffix: str = '_PROGRAMMATIC', hide_original: bool = True, hide_backup: bool = True, allow_non_uniform_scaling: bool = False, aspect_tolerance: float = 0.01, link: bool = False) -> Any:
        """For each layer in ``layer_rects``: optionally duplicate the existing object as a (hidden) dated backup, place a ``*_PROGRAMMATIC`` sibling in the same parent as the source, hide the original, and report actual vs. target bounds.

        Idempotent: existing rows named ``{source}{new_suffix}`` are deleted first so re-runs do not stack copies.
        Group / Clip Group sources (e.g. ``Fig3a.svg`` under ``Fig3a+b``) stay objects in that parent — ``layer.new`` cannot nest a Layer inside a Group.
        Layer sources get a sibling Layer at the same level (e.g. ``C_Data.svg`` under ``Fig3c``). ``file.place`` always drops into the current Layer, so Groups are moved with ``layer.move`` after placing.
        """
        def _subfn_uniform_fit_rect(target: Dict[str, float], source_size: Tuple[float, float]) -> Dict[str, float]:
            sw, sh = source_size
            scale: float = min(target['width'] / sw, target['height'] / sh)
            w, h = sw * scale, sh * scale
            return {'x': target['x'] + (target['width'] - w) / 2.0, 'y': target['y'] + (target['height'] - h) / 2.0, 'width': w, 'height': h}

        def _subfn_inspect_parents() -> Dict[Any, Dict[str, Any]]:
            tree = live.inspect_document(depth=8, childLimit=500)
            return cls._subfn_parent_map(list(tree.get('layers') or []))

        def _subfn_ancestor_has_suffix(node_id: Any, suffix: str, parent_by_id: Dict[Any, Dict[str, Any]]) -> bool:
            seen = set()
            current_id = node_id
            while (current_id is not None) and (current_id not in seen):
                seen.add(current_id)
                parent = parent_by_id.get(current_id) or {}
                if str(parent.get('name') or '').endswith(suffix):
                    return True
                current_id = parent.get('id')
            ## END while (current_id is not None) and (current_id not in seen)...
            return False

        def _subfn_pick_source(matches: List[Dict[str, Any]], parent_by_id: Dict[Any, Dict[str, Any]]) -> List[Dict[str, Any]]:
            not_nested = [m for m in matches if (not _subfn_ancestor_has_suffix(m.get('id'), new_suffix, parent_by_id))]
            chosen = not_nested if not_nested else list(matches)
            return chosen[:1]

        def _subfn_delete_named(name: str, parent_by_id: Dict[Any, Dict[str, Any]]) -> List[Any]:
            matches = live.find_by_name(name, exact=True, limit=200)
            match_ids: List[Any] = [m['id'] for m in matches if isinstance(m, dict) and ('id' in m)]
            match_id_set = set(match_ids)
            root_ids: List[Any] = [node_id for node_id in match_ids if ((parent_by_id.get(node_id) or {}).get('id') not in match_id_set)]
            if root_ids:
                live.execute('layer.delete', {'ids': root_ids})
            return root_ids

        def _subfn_new_sibling_layer(layer_name: str, beside_id: Any) -> Any:
            live.execute('layer.setCurrent', {'id': beside_id})
            reply = live.execute('layer.new', {'name': layer_name, 'visible': True})
            assert isinstance(reply, dict) and ('id' in reply), f"layer.new did not return an id for {layer_name!r} (reply={reply!r})"
            new_id = reply['id']
            live.execute('layer.move', {'ids': [new_id], 'target': beside_id, 'place': 'above'})
            live.execute('layer.setCurrent', {'id': new_id})
            return new_id

        def _subfn_place_as_sibling(new_layer_name: str, replacement_file: Path, place_rect: Dict[str, float], source_id: Optional[Any], source_kind: Optional[str], parent_by_id: Dict[Any, Dict[str, Any]]) -> Tuple[Any, List[Any], Dict[str, float]]:
            if (source_id is not None) and cls._subfn_kind_is_layer(source_kind):
                new_layer_id = _subfn_new_sibling_layer(new_layer_name, source_id)
                placed_ids, actual = live.place_file(replacement_file, place_rect, name=new_layer_name, link=link)
                return new_layer_id, placed_ids, actual
            if source_id is not None:
                parent = parent_by_id.get(source_id) or {}
                current_id = parent.get('id') if (parent.get('id') is not None) else source_id
                live.execute('layer.setCurrent', {'id': current_id})
                placed_ids, actual = live.place_file(replacement_file, place_rect, name=new_layer_name, link=link)
                live.execute('layer.move', {'ids': placed_ids, 'target': source_id, 'place': 'above'})
                return placed_ids[0], placed_ids, actual
            placed_ids, actual = live.place_file(replacement_file, place_rect, name=new_layer_name, link=link)
            return (placed_ids[0] if placed_ids else None), placed_ids, actual

        if expect_title is not None:
            info = live.assert_active_document(expect_title)
        else:
            info = live.active_document_info()
        print(f"active document: {info.get('title')!r}  units={info.get('units')!r}")

        dated: Optional[str] = (backup_suffix if backup_suffix is not None else datetime.now().strftime('%Y-%m-%d')) if backup_layers else None
        search_path: Path = Path(source_folder)
        rows: List[Dict[str, Any]] = []
        for a_layer_name, a_layer_geom in layer_rects.items():
            replacement_file: Path = search_path.joinpath(a_layer_name)
            assert replacement_file.exists(), f"replacement_file: '{replacement_file.as_posix()}' does not exist!"

            parent_by_id = _subfn_inspect_parents()
            new_layer_name: str = f"{a_layer_name}{new_suffix}"
            deleted_ids: List[Any] = _subfn_delete_named(new_layer_name, parent_by_id)
            if deleted_ids:
                parent_by_id = _subfn_inspect_parents()

            existing = live.find_by_name(a_layer_name, exact=True)
            source_matches = _subfn_pick_source(existing, parent_by_id)
            existing_ids: List[Any] = [m['id'] for m in source_matches]
            source_id: Optional[Any] = existing_ids[0] if existing_ids else None
            source_kind: Optional[str] = source_matches[0].get('kind') if source_matches else None
            if (source_id is not None) and (source_kind is None):
                source_kind = (live.get_node(source_id, summary=True, depth=0) or {}).get('kind')
            parent_info = parent_by_id.get(source_id) or {} if (source_id is not None) else {}
            parent_id = parent_info.get('id')
            parent_name = parent_info.get('name')
            if not existing_ids:
                warnings.warn(f"no existing layer named {a_layer_name!r} in {info.get('title')!r}; placing without backup")

            backup_layer_name: Optional[str] = None
            backup_ids: List[Any] = []
            if existing_ids and backup_layers:
                backup_layer_name = live.unique_name(f"{a_layer_name}_{dated}")
                backup_ids = live.duplicate(existing_ids)
                live.execute('object.setProps', {'ids': backup_ids, 'name': backup_layer_name, 'visible': (not hide_backup)})

            source_size = intrinsic_size_pts(replacement_file)
            target_rect: Dict[str, float] = dict(a_layer_geom)
            place_rect: Dict[str, float] = dict(a_layer_geom)
            aspect_ratio_error: Optional[float] = None
            if source_size is not None:
                target_aspect: float = target_rect['width'] / target_rect['height']
                source_aspect: float = source_size[0] / source_size[1]
                aspect_ratio_error = abs(math.log(target_aspect / source_aspect))
                if aspect_ratio_error > aspect_tolerance:
                    msg = (f"{a_layer_name!r}: target aspect {target_aspect:.4f} ({target_rect['width']:.2f}x{target_rect['height']:.2f} pt) != source aspect {source_aspect:.4f} "
                           f"({source_size[0]:.2f}x{source_size[1]:.2f} pt); non-uniform scaling would be required (|log ratio| = {aspect_ratio_error:.4f}).")
                    if allow_non_uniform_scaling:
                        warnings.warn(msg + ' Placing at the exact target rect anyway.')
                    else:
                        place_rect = _subfn_uniform_fit_rect(target_rect, source_size)
                        warnings.warn(msg + f" Placing uniformly-scaled and centered instead: {place_rect}")
            else:
                warnings.warn(f"{a_layer_name!r}: could not read intrinsic size of '{replacement_file.suffix}' file; aspect not checked.")

            new_layer_id, placed_ids, actual = _subfn_place_as_sibling(new_layer_name, replacement_file, place_rect, source_id, source_kind, parent_by_id)
            print(f"  {a_layer_name}: {new_layer_name!r} id={new_layer_id} parent={parent_name!r} placed_ids={placed_ids} deleted_ids={deleted_ids}")
            if hide_original and existing_ids:
                live.execute('object.setProps', {'ids': existing_ids, 'visible': False})

            rows.append({'layer': a_layer_name, 'replacement_file': replacement_file.as_posix(), 'existing_ids': existing_ids, 'source_kind': source_kind, 'parent_id': parent_id, 'parent_name': parent_name,
                         'deleted_ids': deleted_ids, 'backup_layer': backup_layer_name, 'backup_ids': backup_ids,
                         'new_layer': new_layer_name, 'new_layer_id': new_layer_id, 'placed_ids': placed_ids, 'source_w': (source_size or (None, None))[0], 'source_h': (source_size or (None, None))[1], 'aspect_ratio_error': aspect_ratio_error,
                         'target_x': target_rect['x'], 'target_y': target_rect['y'], 'target_w': target_rect['width'], 'target_h': target_rect['height'],
                         'placed_x': place_rect['x'], 'placed_y': place_rect['y'], 'placed_w': place_rect['width'], 'placed_h': place_rect['height'],
                         'actual_x': actual['x'], 'actual_y': actual['y'], 'actual_w': actual['width'], 'actual_h': actual['height'],
                         'dx': actual['x'] - place_rect['x'], 'dy': actual['y'] - place_rect['y'], 'dw': actual['width'] - place_rect['width'], 'dh': actual['height'] - place_rect['height']})
        ## END for a_layer_name, a_layer_geom in layer_rects.items()...

        return pd.DataFrame(rows).set_index('layer', drop=True)


def layer_geometry(live: LiveVectorCraft, layer_names: Sequence[str]) -> Any:
    """Lookup named layers in the active document; return a DataFrame indexed by ``name`` with x/y/width/height."""
    return Layers.layer_geometry(live, layer_names)


def replace_layers(live: LiveVectorCraft, layer_rects: Mapping[str, Mapping[str, float]], source_folder: Union[str, Path], *, expect_title: Optional[str] = None, backup_layers: bool = True, backup_suffix: Optional[str] = None, new_suffix: str = '_PROGRAMMATIC', hide_original: bool = True, hide_backup: bool = True, allow_non_uniform_scaling: bool = False, aspect_tolerance: float = 0.01, link: bool = False) -> Any:
    """For each layer in ``layer_rects``: optionally duplicate the existing object as a (hidden) dated backup, place a ``*_PROGRAMMATIC`` sibling in the same parent as the source, and report actual vs. target bounds."""
    return Layers.replace_layers(live, layer_rects, source_folder, expect_title=expect_title, backup_layers=backup_layers, backup_suffix=backup_suffix, new_suffix=new_suffix, hide_original=hide_original, hide_backup=hide_backup, allow_non_uniform_scaling=allow_non_uniform_scaling, aspect_tolerance=aspect_tolerance, link=link)
