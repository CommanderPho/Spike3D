"""Offline unit tests for origin-point capture, place-rect math, and replace_by_origin."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from vectorcraft_scripting.origin import Origin, OriginPoint, align_by_origin, place_rect_for_origin, replace_by_origin


def _path_node(anchors: List[Tuple[float, float]], *, name: str = '<Path>') -> Dict[str, Any]:
    return {
        'name': name,
        'kind': {'type': 'path', 'rule': 'EvenOdd', 'path': {'subpaths': [{'closed': False, 'anchors': [{'p': [x, y]} for x, y in anchors]}]}},
        'bounds': {'x': min(a[0] for a in anchors), 'y': min(a[1] for a in anchors), 'width': max(a[0] for a in anchors) - min(a[0] for a in anchors), 'height': max(a[1] for a in anchors) - min(a[1] for a in anchors)},
    }


class FakeOriginLive:
    """Live stand-in that supports Direct-Selected anchor probing via path.moveAnchors + undo."""

    def __init__(self, nodes: Dict[Any, Dict[str, Any]], selection: List[Any], *, selected_anchors: Optional[Dict[Any, List[Tuple[int, int]]]] = None, title: str = 'Fig.pdf'):
        self.nodes = {nid: dict(node) for nid, node in nodes.items()}
        self.selection = list(selection)
        self.selected_anchors = {nid: list(refs) for nid, refs in (selected_anchors or {}).items()}
        self.title = title
        self.executed: List[Any] = []
        self._probe_active = False
        self._probe_dx = 0.0


    def active_document_info(self) -> Dict[str, Any]:
        return {'title': self.title}


    def inspect_document(self, **params: Any) -> Dict[str, Any]:
        return {'selection': list(self.selection), 'title': self.title, 'selectionBounds': self.node_bounds(self.selection[0]) if self.selection else None}


    def get_node(self, node_id: Any, *, summary: bool = True, depth: Optional[int] = None, child_limit: Optional[int] = None) -> Dict[str, Any]:
        node = dict(self.nodes[node_id])
        node['id'] = node_id
        if summary:
            node['kind'] = 'Path'
        return node


    def node_bounds(self, node_id: Any) -> Dict[str, float]:
        return dict(self.nodes[node_id]['bounds'])


    def select_document(self, name: str) -> Dict[str, Any]:
        assert name == self.title
        return {'title': self.title}


    def _anchor_xy(self, node_id: Any, si: int, ai: int) -> List[float]:
        return self.nodes[node_id]['kind']['path']['subpaths'][si]['anchors'][ai]['p']


    def execute(self, command: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(params or {})
        self.executed.append((command, params))
        if command == 'path.moveAnchors':
            dx = float(params.get('dx') or 0.0)
            dy = float(params.get('dy') or 0.0)
            self._probe_active = True
            self._probe_dx = dx
            # Direct-selected anchors may live on child paths while the selection lists a parent group.
            for nid, refs in self.selected_anchors.items():
                for si, ai in refs:
                    p = self._anchor_xy(nid, si, ai)
                    p[0] += dx
                    p[1] += dy
                ## END for si, ai in refs...
            ## END for nid, refs in self.selected_anchors.items()...
            return None
        if command == 'edit.undo':
            if self._probe_active:
                for nid, refs in self.selected_anchors.items():
                    for si, ai in refs:
                        p = self._anchor_xy(nid, si, ai)
                        p[0] -= self._probe_dx
                    ## END for si, ai in refs...
                ## END for nid, refs in self.selected_anchors.items()...
                self._probe_active = False
                self._probe_dx = 0.0
            return None
        if command == 'object.move':
            dx = float(params['dx'])
            dy = float(params['dy'])
            for nid in params['ids']:
                path = self.nodes[nid]['kind']['path']
                for sp in path['subpaths']:
                    for anchor in sp['anchors']:
                        anchor['p'][0] += dx
                        anchor['p'][1] += dy
                    ## END for anchor in sp['anchors']...
                ## END for sp in path['subpaths']...
                b = self.nodes[nid]['bounds']
                b['x'] += dx
                b['y'] += dy
            ## END for nid in params['ids']...
            return None
        if command == 'object.setProps':
            for nid in params.get('ids') or []:
                self.nodes[nid]['visible'] = params.get('visible', self.nodes[nid].get('visible'))
            ## END for nid in params.get('ids') or []...
            return None
        if command == 'select.set':
            self.selection = list(params.get('ids') or [])
            return None
        return {'ok': True}


def test_place_rect_for_origin_scales_and_offsets():
    rect = place_rect_for_origin(target_xy=(100.0, 200.0), source_origin_xy=(10.0, 20.0), source_size=(50.0, 40.0), scale=2.0)
    assert rect == {'x': 80.0, 'y': 160.0, 'width': 100.0, 'height': 80.0}


def test_capture_from_selection_uses_selected_anchor_centroid():
    live = FakeOriginLive(
        nodes={7: _path_node([(10.0, 30.0), (40.0, 30.0)])},
        selection=[7],
        selected_anchors={7: [(0, 0)]},
    )
    origin = Origin.capture_from_selection(live)
    assert origin.xy == (10.0, 30.0)
    assert origin.object_ids == [7]
    assert origin.anchor_refs == {7: [(0, 0)]}
    assert origin.fully_selected is False
    assert origin.document_title == 'Fig.pdf'


def test_capture_from_selection_marks_fully_selected_path():
    live = FakeOriginLive(
        nodes={7: _path_node([(10.0, 30.0), (40.0, 30.0)])},
        selection=[7],
        selected_anchors={7: [(0, 0), (0, 1)]},
    )
    origin = Origin.capture_from_selection(live)
    assert origin.xy == (25.0, 30.0)
    assert origin.fully_selected is True


def test_replace_by_origin_moves_source_and_hides_target():
    live = FakeOriginLive(
        nodes={
            1: _path_node([(100.0, 50.0), (130.0, 50.0)], name='old'),
            2: _path_node([(10.0, 80.0), (40.0, 80.0)], name='new'),
        },
        selection=[2],
        selected_anchors={2: [(0, 0)]},
        title='Fig.pdf',
    )
    target = OriginPoint(x=100.0, y=50.0, object_ids=[1], document_title='Fig.pdf')
    source = OriginPoint(x=10.0, y=80.0, object_ids=[2], document_title='Fig.pdf')
    result = replace_by_origin(live, target, source, hide_target=True)
    assert result['align']['dx'] == pytest.approx(90.0)
    assert result['align']['dy'] == pytest.approx(-30.0)
    assert live.nodes[2]['kind']['path']['subpaths'][0]['anchors'][0]['p'] == [100.0, 50.0]
    assert live.nodes[1].get('visible') is False
    assert live.selection == [2]


def test_align_by_origin_no_op_when_already_aligned():
    live = FakeOriginLive(nodes={2: _path_node([(5.0, 5.0)])}, selection=[2], selected_anchors={2: [(0, 0)]})
    target = OriginPoint.from_xy(5.0, 5.0, document_title='Fig.pdf')
    source = OriginPoint(x=5.0, y=5.0, object_ids=[2], document_title='Fig.pdf')
    result = align_by_origin(live, target, source)
    assert result['dx'] == 0.0 and result['dy'] == 0.0
    assert ('object.move', {'ids': [2], 'dx': 0.0, 'dy': 0.0}) not in [(c, p) for c, p in live.executed if c == 'object.move']


def test_capture_from_selection_walks_group_children():
    child = _path_node([(10.0, 30.0), (40.0, 30.0)], name='line')
    group = {
        'name': 'panel',
        'kind': {'type': 'group', 'children': [{**child, 'id': 8}]},
        'bounds': {'x': 10.0, 'y': 30.0, 'width': 30.0, 'height': 0.0},
    }
    live = FakeOriginLive(nodes={7: group, 8: child}, selection=[7], selected_anchors={8: [(0, 0)]})
    origin = Origin.capture_from_selection(live)
    assert origin.xy == (10.0, 30.0)
    assert origin.object_ids == [7]
    assert origin.anchor_refs == {8: [(0, 0)]}
    assert origin.fully_selected is False
