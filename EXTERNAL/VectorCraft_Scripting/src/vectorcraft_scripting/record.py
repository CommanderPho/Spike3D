"""Record a series of changes to objects, then replay them onto similar objects via script.

Two complementary routes:

- Route A (``Actions``): wrap VectorCraft's native Actions panel recorder for edits made by hand
  in the GUI. Recorded sets are saved as ``.vcactions`` files and played back with ``ui.action.play``
  (accelerated = one undoable batch). Recorded steps are selection/id-relative: select the similar
  target object first (``select_ids``) before playing.
- Route B (``CommandRecorder`` + ``replay_recipe``): capture every mutating ``live.execute(...)`` call
  made from Python (including ``TextHelpers`` / ``Layers`` helpers) into a JSON recipe. Object
  references are stored as name/kind/text/bounds descriptors instead of raw ids, so the recipe can be
  replayed onto similar objects in the same or another document. Replay resolves descriptors, remaps
  ids, and applies everything through ``command.batch`` (ONE undo step, full rollback on error).

Caveats:
- Geometry-derived deltas (e.g. the corrective ``object.move`` inside ``set_text_font_size_centered``)
  replay exactly as recorded — correct for same-size similar objects, not re-derived for other sizes.
- Recipes that create objects (``file.place``, ``layer.new``, …) reference ids that only exist in the
  source document; replaying those steps onto another document needs an explicit ``id_map``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from .find import ObjectQuery, bounds_xywh
from .live import LiveVectorCraft

## Commands that never mutate the document: passed through without being recorded.
READ_ONLY_COMMANDS = {'document.find', 'document.node', 'document.inspect', 'document.json', 'document.serialize', 'document.formats', 'document.pdfInfo', 'document.exportSettings', 'document.compare', 'text.getRange', 'file.info', 'file.list', 'file.recentFiles', 'engine.commands', 'ui.inspect', 'clipboard.flavours', 'clipboard.conflicts'}
READ_ONLY_PREFIXES = ('ui.', 'view.', 'window.', 'tool.', 'help.', 'app.', 'script.')
READ_ONLY_SUFFIXES = ('.list',)

## Commands that act on the current selection when given no explicit ids.
SELECTION_DEPENDENT_COMMANDS = {'edit.cut', 'edit.copy', 'edit.paste', 'edit.pasteInFront', 'edit.pasteInBack', 'edit.pasteInPlace', 'edit.pasteOnAllArtboards', 'edit.clear', 'edit.duplicate', 'object.group', 'object.ungroup', 'object.hide', 'object.showAll', 'object.lock', 'object.unlockAll', 'object.align', 'object.distribute', 'object.distributeSpacing', 'transparency.set', 'type.createOutlines'}
SELECTION_DEPENDENT_PREFIXES = ('paint.', 'stroke.', 'object.arrange.')

## Params keys whose values are object ids (remapped at replay; never value-matched elsewhere).
ID_PARAM_KEYS = ('ids', 'id', 'target')

## Undo-group markers are dropped at replay: command.batch already provides the single undo step.
REPLAY_DROPPED_COMMANDS = {'edit.undoGroup.begin', 'edit.undoGroup.end'}

RECIPE_VERSION = 1


def _is_read_only_command(command: str) -> bool:
    return (command in READ_ONLY_COMMANDS) or command.startswith(READ_ONLY_PREFIXES) or command.endswith(READ_ONLY_SUFFIXES)


def _is_selection_dependent(command: str) -> bool:
    return (command in SELECTION_DEPENDENT_COMMANDS) or command.startswith(SELECTION_DEPENDENT_PREFIXES)


class Actions:
    """Route A: record GUI edits with VectorCraft's native Actions panel and play them back."""

    @classmethod
    def start(cls, live: LiveVectorCraft, name: Optional[str] = None, set_index: int = 0) -> Dict[str, Any]:
        """Begin recording: ``ui.action.new`` (named) or ``ui.action.record`` (auto-named Action N). Returns handle {set, action, name}."""
        if name is not None:
            reply = live.execute('ui.action.new', {'set': int(set_index), 'name': name})
            assert isinstance(reply, dict) and ('action' in reply), f"ui.action.new did not return an action index (reply={reply!r})"
            return {'set': reply.get('set', set_index), 'action': reply['action'], 'name': name}
        reply = live.execute('ui.action.record', {'set': int(set_index)})
        assert isinstance(reply, dict), f"ui.action.record did not return a dict (reply={reply!r})"
        return {'set': reply.get('set', set_index), 'action': None, 'name': reply.get('name')}


    @classmethod
    def stop(cls, live: LiveVectorCraft) -> Any:
        """End the recording and keep its steps (``ui.action.stop``)."""
        return live.execute('ui.action.stop', {})


    @classmethod
    @contextmanager
    def record_action(cls, live: LiveVectorCraft, name: Optional[str] = None, set_index: int = 0) -> Iterator[Dict[str, Any]]:
        """Record GUI edits made inside the block; recording always stops on exit."""
        handle = cls.start(live, name=name, set_index=set_index)
        try:
            yield handle
        finally:
            cls.stop(live)


    @classmethod
    def save(cls, live: LiveVectorCraft, path: Union[str, Path], set_index: int = 0) -> Path:
        """Write the Actions panel set (with all its actions) to a ``.vcactions`` file (``ui.actionSet.save``)."""
        dest: Path = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        live.execute('ui.actionSet.save', {'set': int(set_index), 'path': str(dest)})
        assert dest.exists(), f"ui.actionSet.save did not write '{dest.as_posix()}'"
        return dest


    @classmethod
    def load(cls, live: LiveVectorCraft, path: Union[str, Path]) -> Any:
        """Load a ``.vcactions`` file into the Actions panel (``ui.actionSet.load``); returns {set, name, actions}."""
        src: Path = Path(path).resolve()
        assert src.exists(), f"action set file does not exist: '{src.as_posix()}'"
        return live.execute('ui.actionSet.load', {'path': str(src)})


    @classmethod
    def play(cls, live: LiveVectorCraft, action: str, target_document: Optional[str] = None, select_ids: Optional[Sequence[Any]] = None, set_name: Optional[str] = None) -> Any:
        """Play a recorded action (accelerated = one undoable batch).

        Recorded steps are selection/id-relative: pass ``target_document`` to activate the right tab
        and ``select_ids`` to select the similar target object before playback.
        """
        if target_document is not None:
            live.select_document(target_document)
            live.assert_active_document(target_document)
        if select_ids is not None:
            live.execute('select.set', {'ids': list(select_ids)})
        live.execute('ui.action.playbackOptions', {'speed': 'accelerated'})
        params: Dict[str, Any] = {'action': action}
        if set_name is not None:
            params['set'] = set_name
        return live.execute('ui.action.play', params)


    @classmethod
    def batch(cls, live: LiveVectorCraft, commands: Sequence[Mapping[str, Any]], folder: Union[str, Path], destination: str = 'none', destination_folder: Optional[Union[str, Path]] = None, include_subdirs: bool = False, override_open: bool = True, override_save: bool = False, override_export: bool = False, stop_on_error: bool = True, label: Optional[str] = None) -> Any:
        """Play explicit steps on every readable file in ``folder`` (``action.batch``).

        The engine's ``action.batch`` takes the steps themselves (not an action name): pass a
        recipe's ``steps`` (or steps parsed from a saved action set). ``destination`` is
        ``none`` (leave open) | ``saveAndClose`` | ``folder`` (needs ``destination_folder``).
        """
        src: Path = Path(folder).resolve()
        assert src.is_dir(), f"batch source folder does not exist: '{src.as_posix()}'"
        params: Dict[str, Any] = {'commands': [dict(c) for c in commands], 'source': 'folder', 'folder': str(src), 'includeSubdirs': bool(include_subdirs), 'destination': destination, 'overrideOpen': bool(override_open), 'overrideSave': bool(override_save), 'overrideExport': bool(override_export), 'stopOnError': bool(stop_on_error)}
        if destination_folder is not None:
            params['destinationFolder'] = str(Path(destination_folder).resolve())
        if label is not None:
            params['label'] = label
        return live.execute('action.batch', params)


class CommandRecorder:
    """Route B: record mutating ``live.execute(...)`` calls into an id-independent JSON recipe.

    Usage::

        with CommandRecorder(live) as rec:
            TextHelpers.set_text_font_size_centered(live, label_ids, size_pt=8)
        save_recipe(rec.recipe, 'bump_labels.json')
        replay_recipe(live, load_recipe('bump_labels.json'), target_document='Other Figure.pdf')

    Wraps ``live.execute`` at instance level for the duration of the block (restored on exit), so
    every existing helper that goes through ``live.execute`` is captured automatically. Descriptors
    are captured BEFORE each step runs, so renames/retext steps still resolve by the original identity.
    """

    def __init__(self, live: LiveVectorCraft):
        self.live = live
        self.steps: List[Dict[str, Any]] = []
        self.objects: Dict[str, Dict[str, Any]] = {}  # str(old_id) -> {name, kind, bounds, text?}
        self.source_document: Optional[str] = None
        self.recorded_at: Optional[str] = None
        self._recipe: Optional[Dict[str, Any]] = None
        self._original_execute = None
        self._had_instance_execute: bool = False
        self._last_selected_ids: Optional[List[Any]] = None


    def __enter__(self) -> "CommandRecorder":
        info = self.live.active_document_info()
        self.source_document = str(info.get('title') or '') if isinstance(info, dict) else None
        self.recorded_at = datetime.now().isoformat(timespec='seconds')
        self._had_instance_execute = 'execute' in vars(self.live)
        self._original_execute = self.live.execute
        self.live.execute = self._execute_and_record
        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        if self._had_instance_execute:
            self.live.execute = self._original_execute
        else:
            del self.live.execute
        self._recipe = {'version': RECIPE_VERSION, 'source_document': self.source_document, 'recorded_at': self.recorded_at, 'objects': dict(self.objects), 'steps': list(self.steps)}


    @property
    def recipe(self) -> Dict[str, Any]:
        """The finished recipe; available after the ``with`` block exits."""
        assert self._recipe is not None, "recipe is only available after the CommandRecorder with-block exits"
        return self._recipe


    def _execute_and_record(self, command: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        if _is_read_only_command(command):
            return self._original_execute(command, params)
        step_params: Dict[str, Any] = deepcopy(dict(params)) if params is not None else {}
        if _is_selection_dependent(command) and not any(k in step_params for k in ('ids', 'id')):
            self._record_selection_step()
        self._capture_referenced_ids(step_params)
        self.steps.append({'command': command, 'params': step_params})
        if command in ('select.set', 'select.add', 'select.toggle'):
            self._last_selected_ids = list(step_params.get('ids') or [])
        return self._original_execute(command, params)


    def _record_selection_step(self) -> None:
        """Snapshot the current selection and prepend a ``select.set`` step when it changed."""
        tree = self.live.inspect_document(depth=0, childLimit=0)
        selection: List[Any] = list(tree.get('selection') or []) if isinstance(tree, dict) else []
        if not selection or (selection == self._last_selected_ids):
            return
        self._capture_referenced_ids({'ids': selection})
        self.steps.append({'command': 'select.set', 'params': {'ids': list(selection)}})
        self._last_selected_ids = list(selection)


    def _capture_referenced_ids(self, params: Mapping[str, Any]) -> None:
        for key in ID_PARAM_KEYS:
            if key not in params:
                continue
            value = params[key]
            ids: List[Any] = list(value) if (key == 'ids') and isinstance(value, (list, tuple)) else [value]
            for node_id in ids:
                if node_id is None:
                    continue
                self._capture_descriptor(node_id)
            ## END for node_id in ids...
        ## END for key in ID_PARAM_KEYS...


    def _capture_descriptor(self, node_id: Any) -> None:
        key: str = str(node_id)
        if key in self.objects:
            return
        node = self.live.get_node(node_id, summary=True, depth=0)
        assert isinstance(node, dict), f"document.node did not return a dict for id={node_id!r}: {node!r}"
        descriptor: Dict[str, Any] = {'name': node.get('name'), 'kind': node.get('kind'), 'bounds': bounds_xywh(node.get('bounds'))}
        if str(node.get('kind') or '') == 'Type':
            rng = self._original_execute('text.getRange', {'id': node_id, 'effective': True})
            descriptor['text'] = (rng or {}).get('text') if isinstance(rng, dict) else None
        self.objects[key] = descriptor


def save_recipe(recipe: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Write a recipe dict to a JSON file; returns the resolved path."""
    dest: Path = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding='utf-8')
    return dest


def load_recipe(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a recipe JSON file; object keys are normalized back to ints when they look like ints."""
    src: Path = Path(path).resolve()
    recipe = json.loads(src.read_text(encoding='utf-8'))
    assert isinstance(recipe, dict) and (recipe.get('version') == RECIPE_VERSION), f"unsupported recipe version in '{src.as_posix()}': {recipe.get('version') if isinstance(recipe, dict) else recipe!r}"
    objects: Dict[str, Dict[str, Any]] = {}
    for key, descriptor in (recipe.get('objects') or {}).items():
        objects[str(key)] = descriptor
    ## END for key, descriptor in (recipe.get('objects') or {}).items()...

    recipe['objects'] = objects
    return recipe


def replay_recipe(live: LiveVectorCraft, recipe: Mapping[str, Any], target_document: Optional[str] = None, id_map: Optional[Mapping[Any, Any]] = None, backup: bool = True, label: Optional[str] = None) -> Any:
    """Apply a recorded recipe to similar objects in the active (or ``target_document``) document.

    Resolution order per referenced object: explicit ``id_map`` entry, else exact name (+kind) match,
    else for Type objects a text match disambiguated by nearest recorded bounds. Every descriptor must
    resolve to exactly one object (loud failure otherwise — pass ``id_map`` for ambiguous cases).
    Steps run through ``command.batch``: ONE undo step, full rollback on the first error.
    """
    assert recipe.get('version') == RECIPE_VERSION, f"unsupported recipe version: {recipe.get('version')!r}"
    if target_document is not None:
        live.select_document(target_document)
        live.assert_active_document(target_document)

    explicit_map: Dict[str, Any] = {str(k): v for k, v in (id_map or {}).items()}
    objects: Dict[str, Dict[str, Any]] = dict(recipe.get('objects') or {})

    def _subfn_resolve_descriptor(old_id: str, descriptor: Mapping[str, Any]) -> Any:
        if old_id in explicit_map:
            return explicit_map[old_id]
        name = descriptor.get('name')
        kind = descriptor.get('kind')
        candidates: List[Dict[str, Any]] = []
        if name:
            candidates = live.find_by_name(str(name), exact=True)
            if kind:
                candidates = [m for m in candidates if str(m.get('kind') if ('kind' in m) else (live.get_node(m['id'], summary=True, depth=0) or {}).get('kind')) == str(kind)]
        if len(candidates) == 1:
            return candidates[0]['id']
        assert len(candidates) <= 1, f"descriptor for old id {old_id} ({name!r}, kind={kind!r}) matches {len(candidates)} objects {[m.get('id') for m in candidates]} — pass an explicit id_map entry"
        if (str(kind or '') == 'Type') and (descriptor.get('text') is not None):
            text_matches = live.find_objects(ObjectQuery(kind='Type', text=str(descriptor['text']), limit=1000))
            if len(text_matches) == 1:
                return text_matches[0]['id']
            recorded_bounds = descriptor.get('bounds')
            if text_matches and recorded_bounds:
                rcx = recorded_bounds['x'] + recorded_bounds['width'] / 2.0
                rcy = recorded_bounds['y'] + recorded_bounds['height'] / 2.0
                distances: List[float] = []
                for m in text_matches:
                    xywh = bounds_xywh(m.get('bounds')) or bounds_xywh((live.get_node(m['id'], summary=True, depth=0) or {}).get('bounds'))
                    assert xywh is not None, f"text match id={m['id']!r} has no bounds to disambiguate old id {old_id}"
                    distances.append(((xywh['x'] + xywh['width'] / 2.0) - rcx) ** 2 + ((xywh['y'] + xywh['height'] / 2.0) - rcy) ** 2)
                ## END for m in text_matches...

                best = min(distances)
                best_ids = [m['id'] for m, d in zip(text_matches, distances) if d == best]
                assert len(best_ids) == 1, f"descriptor for old id {old_id} (text={descriptor.get('text')!r}) has {len(best_ids)} equidistant matches {best_ids} — pass an explicit id_map entry"
                return best_ids[0]
        raise AssertionError(f"could not resolve descriptor for old id {old_id} ({name!r}, kind={kind!r}, text={descriptor.get('text')!r}) in the target document — pass an explicit id_map entry")

    resolved: Dict[str, Any] = {old_id: _subfn_resolve_descriptor(old_id, descriptor) for old_id, descriptor in objects.items()}

    def _subfn_remap(value: Any, key: Optional[str] = None) -> Any:
        if isinstance(value, dict):
            return {k: _subfn_remap(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_subfn_remap(v, key) for v in value]
        if (key in ID_PARAM_KEYS) and (value is not None) and isinstance(value, (int, str)):
            assert str(value) in resolved, f"step param {key}={value!r} was not captured as an object descriptor — pass an explicit id_map entry"
            return resolved[str(value)]
        return value

    commands: List[Dict[str, Any]] = []
    for step in (recipe.get('steps') or []):
        if step['command'] in REPLAY_DROPPED_COMMANDS:
            continue
        commands.append({'command': step['command'], 'params': _subfn_remap(dict(step.get('params') or {}))})
    ## END for step in (recipe.get('steps') or [])...

    assert commands, "recipe contains no replayable steps"
    if backup:
        live.backup_document(expect_title=target_document)
    batch_label: str = label if label is not None else f"Replay recorded changes ({len(commands)} steps)"
    return live.execute('command.batch', {'label': batch_label, 'commands': commands})


def record_action(live: LiveVectorCraft, name: Optional[str] = None, set_index: int = 0) -> Iterator[Dict[str, Any]]:
    """Context manager: record GUI edits into a native action (Route A)."""
    return Actions.record_action(live, name=name, set_index=set_index)


def play_action(live: LiveVectorCraft, action: str, target_document: Optional[str] = None, select_ids: Optional[Sequence[Any]] = None, set_name: Optional[str] = None) -> Any:
    """Play a recorded native action onto the active (or target) document (Route A)."""
    return Actions.play(live, action, target_document=target_document, select_ids=select_ids, set_name=set_name)
