"""Offline unit tests for record/replay (Actions wrapper, CommandRecorder, recipe replay)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from vectorcraft_scripting.find import coerce_query
from vectorcraft_scripting.record import Actions, CommandRecorder, load_recipe, replay_recipe, save_recipe


def _type_node(name: Optional[str], text: str, x: float = 0.0, y: float = 0.0) -> Dict[str, Any]:
    return {'name': name, 'kind': 'Type', 'bounds': {'x': x, 'y': y, 'width': 20.0, 'height': 6.0}, 'text': text}


class FakeLive:
    """Minimal LiveVectorCraft stand-in: in-memory nodes, selection, and an executed-command log."""

    def __init__(self, nodes: Optional[Dict[Any, Dict[str, Any]]] = None, selection: Optional[List[Any]] = None, title: str = 'Source.pdf'):
        self.nodes: Dict[Any, Dict[str, Any]] = dict(nodes or {})
        self.selection: List[Any] = list(selection or [])
        self.title = title
        self.executed: List[Any] = []
        self.backups: List[Any] = []
        self.selected_documents: List[str] = []


    def active_document_info(self) -> Dict[str, Any]:
        return {'title': self.title}


    def execute(self, command: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self.executed.append((command, dict(params or {})))
        if command == 'text.getRange':
            return {'text': self.nodes[params['id']].get('text'), 'runs': []}
        if command == 'ui.action.new':
            return {'set': params.get('set', 0), 'action': 0}
        if command == 'ui.action.record':
            return {'set': params.get('set', 0), 'name': 'Action 1'}
        if command == 'command.batch':
            return {'results': [None] * len(params['commands'])}
        return {'ok': True}


    def get_node(self, node_id: Any, *, summary: bool = True, depth: Optional[int] = None, child_limit: Optional[int] = None) -> Dict[str, Any]:
        node = dict(self.nodes[node_id])
        node.setdefault('id', node_id)
        return node


    def inspect_document(self, **params: Any) -> Dict[str, Any]:
        return {'selection': list(self.selection), 'title': self.title}


    def find_by_name(self, name: str, exact: bool = True, limit: int = 100) -> List[Dict[str, Any]]:
        return [{'id': nid, 'name': n.get('name'), 'kind': n.get('kind'), 'bounds': n.get('bounds')} for nid, n in self.nodes.items() if n.get('name') == name]


    def find_objects(self, query: Any) -> List[Dict[str, Any]]:
        q = coerce_query(query)
        out: List[Dict[str, Any]] = []
        for nid, n in self.nodes.items():
            if (q.kind is not None) and (n.get('kind') != q.kind):
                continue
            if (q.text is not None) and (n.get('text') != q.text):
                continue
            out.append({'id': nid, 'name': n.get('name'), 'kind': n.get('kind'), 'bounds': n.get('bounds')})
        ## END for nid, n in self.nodes.items()...

        return out


    def select_document(self, name: str) -> None:
        self.selected_documents.append(name)


    def assert_active_document(self, expected: str) -> Dict[str, Any]:
        assert self.title == expected, f"active document is {self.title!r}, expected {expected!r}"
        return {'title': self.title}


    def backup_document(self, expect_title: Optional[str] = None, **kwargs: Any) -> Dict[str, Path]:
        self.backups.append(expect_title)
        return {'same_format': Path('bak.pdf')}


def test_recorder_skips_read_only_commands():
    live = FakeLive(nodes={1: _type_node('tick1', '0.0')})
    with CommandRecorder(live) as rec:
        live.execute('document.find', {'kind': 'Type', 'limit': 10})
        live.execute('text.getRange', {'id': 1, 'effective': True})
        live.execute('file.info', {})
        live.execute('swatch.list', {})
        live.execute('ui.inspect', {})
        live.get_node(1)

    assert rec.recipe['steps'] == []
    assert rec.recipe['objects'] == {}


def test_recorder_captures_steps_and_descriptors():
    live = FakeLive(nodes={174: _type_node('tick1', '0.0', x=10.0, y=20.0)})
    with CommandRecorder(live) as rec:
        live.execute('edit.undoGroup.begin', {'label': 'Bump'})
        live.execute('text.setStyle', {'ids': [174], 'size': 8.0})
        live.execute('object.move', {'ids': [174], 'dx': 1.0, 'dy': -0.5})
        live.execute('edit.undoGroup.end', {})

    recipe = rec.recipe
    assert [s['command'] for s in recipe['steps']] == ['edit.undoGroup.begin', 'text.setStyle', 'object.move', 'edit.undoGroup.end']
    assert recipe['source_document'] == 'Source.pdf'
    assert recipe['objects']['174']['name'] == 'tick1'
    assert recipe['objects']['174']['kind'] == 'Type'
    assert recipe['objects']['174']['text'] == '0.0'
    assert recipe['objects']['174']['bounds'] == {'x': 10.0, 'y': 20.0, 'width': 20.0, 'height': 6.0}
    assert 'execute' not in vars(live)  # instance-level wrapper removed on exit


def test_recorder_restores_execute_on_exception():
    live = FakeLive(nodes={1: _type_node('tick1', '0.0')})
    with pytest.raises(RuntimeError):
        with CommandRecorder(live) as rec:
            live.execute('text.setStyle', {'ids': [1], 'size': 8.0})
            raise RuntimeError('boom')

    assert 'execute' not in vars(live)
    assert len(rec.recipe['steps']) == 1  # steps recorded before the failure are kept


def test_recorder_inserts_selection_step_once():
    live = FakeLive(nodes={5: {'name': 'p1', 'kind': 'Path', 'bounds': None}, 9: {'name': 'p2', 'kind': 'Path', 'bounds': None}}, selection=[5, 9])
    with CommandRecorder(live) as rec:
        live.execute('paint.setFill', {'color': '#ff0000'})
        live.execute('paint.setFill', {'color': '#00ff00'})
        live.execute('object.setProps', {'ids': [5], 'name': 'renamed'})

    commands = [s['command'] for s in rec.recipe['steps']]
    assert commands == ['select.set', 'paint.setFill', 'paint.setFill', 'object.setProps']
    assert rec.recipe['steps'][0]['params'] == {'ids': [5, 9]}
    assert set(rec.recipe['objects']) == {'5', '9'}


def test_recipe_round_trip(tmp_path: Path):
    live = FakeLive(nodes={174: _type_node('tick1', '0.0')})
    with CommandRecorder(live) as rec:
        live.execute('text.setStyle', {'ids': [174], 'size': 8.0})

    path = save_recipe(rec.recipe, tmp_path / 'recipe.json')
    loaded = load_recipe(path)
    assert loaded == rec.recipe
    assert loaded['objects']['174']['text'] == '0.0'


def test_replay_remaps_ids_drops_undo_groups_and_batches():
    source = FakeLive(nodes={174: _type_node('tick1', '0.0')})
    with CommandRecorder(source) as rec:
        source.execute('edit.undoGroup.begin', {'label': 'Bump'})
        source.execute('text.setStyle', {'ids': [174], 'size': 8.0})
        source.execute('object.move', {'ids': [174], 'dx': 1.0, 'dy': -0.5})
        source.execute('edit.undoGroup.end', {})

    target = FakeLive(nodes={900: _type_node('tick1', '0.0', x=50.0)}, title='Target.pdf')
    replay_recipe(target, rec.recipe, target_document='Target.pdf', backup=True)

    assert target.selected_documents == ['Target.pdf']
    assert target.backups == ['Target.pdf']
    batch_commands = [c for cmd, c in target.executed if cmd == 'command.batch']
    assert len(batch_commands) == 1
    steps = batch_commands[0]['commands']
    assert [s['command'] for s in steps] == ['text.setStyle', 'object.move']
    assert steps[0]['params'] == {'ids': [900], 'size': 8.0}
    assert steps[1]['params'] == {'ids': [900], 'dx': 1.0, 'dy': -0.5}


def test_replay_does_not_value_match_non_id_params():
    source = FakeLive(nodes={8: _type_node('tick1', '0.0')})
    with CommandRecorder(source) as rec:
        source.execute('text.setStyle', {'ids': [8], 'size': 8.0})  # size equals the old id on purpose

    target = FakeLive(nodes={900: _type_node('tick1', '0.0')}, title='Target.pdf')
    replay_recipe(target, rec.recipe, backup=False)
    steps = [c for cmd, c in target.executed if cmd == 'command.batch'][0]['commands']
    assert steps[0]['params'] == {'ids': [900], 'size': 8.0}


def test_replay_id_map_overrides_resolution():
    source = FakeLive(nodes={174: _type_node('tick1', '0.0')})
    with CommandRecorder(source) as rec:
        source.execute('text.setStyle', {'ids': [174], 'size': 8.0})

    target = FakeLive(nodes={900: _type_node('totally_different_name', '0.0')}, title='Target.pdf')
    replay_recipe(target, rec.recipe, id_map={174: 900}, backup=False)
    steps = [c for cmd, c in target.executed if cmd == 'command.batch'][0]['commands']
    assert steps[0]['params']['ids'] == [900]


def test_replay_ambiguous_name_raises():
    source = FakeLive(nodes={174: _type_node('tick1', '0.0')})
    with CommandRecorder(source) as rec:
        source.execute('text.setStyle', {'ids': [174], 'size': 8.0})

    target = FakeLive(nodes={900: _type_node('tick1', '0.0'), 901: _type_node('tick1', '1.0')}, title='Target.pdf')
    with pytest.raises(AssertionError, match='id_map'):
        replay_recipe(target, rec.recipe, backup=False)


def test_replay_unnamed_type_resolves_by_text_and_nearest_bounds():
    source = FakeLive(nodes={174: _type_node(None, '0.0', x=100.0, y=100.0)})
    with CommandRecorder(source) as rec:
        source.execute('text.setStyle', {'ids': [174], 'size': 8.0})

    target = FakeLive(nodes={900: _type_node(None, '0.0', x=10.0, y=10.0), 901: _type_node(None, '0.0', x=99.0, y=101.0)}, title='Target.pdf')
    replay_recipe(target, rec.recipe, backup=False)
    steps = [c for cmd, c in target.executed if cmd == 'command.batch'][0]['commands']
    assert steps[0]['params']['ids'] == [901]  # nearest to the recorded bounds center


def test_replay_uncaptured_id_fails_loudly():
    recipe = {'version': 1, 'source_document': 'Source.pdf', 'recorded_at': 'now', 'objects': {}, 'steps': [{'command': 'text.setStyle', 'params': {'ids': [999], 'size': 8.0}}]}
    target = FakeLive(nodes={}, title='Target.pdf')
    with pytest.raises(AssertionError, match='id_map'):
        replay_recipe(target, recipe, backup=False)


def test_actions_wrapper_start_stop_save_play(tmp_path: Path):
    live = FakeLive()
    handle = Actions.start(live, name='Bump labels', set_index=0)
    assert handle['name'] == 'Bump labels'
    Actions.stop(live)

    vca_path = tmp_path / 'bump.vcactions'
    vca_path.write_text('stub', encoding='utf-8')  # the fake app does not write files itself
    saved = Actions.save(live, vca_path, set_index=0)
    assert saved == vca_path.resolve()

    Actions.play(live, action='Bump labels', target_document='Source.pdf', select_ids=[42])

    commands = [cmd for cmd, _ in live.executed]
    assert commands == ['ui.action.new', 'ui.action.stop', 'ui.actionSet.save', 'select.set', 'ui.action.playbackOptions', 'ui.action.play']
    play_params = live.executed[-1][1]
    assert play_params == {'action': 'Bump labels'}
    assert live.executed[3][1] == {'ids': [42]}


def test_actions_record_action_context_stops_on_exception():
    live = FakeLive()
    with pytest.raises(RuntimeError):
        with Actions.record_action(live, name='X'):
            raise RuntimeError('boom')

    commands = [cmd for cmd, _ in live.executed]
    assert commands == ['ui.action.new', 'ui.action.stop']
