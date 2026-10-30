"""High-level live client for a running VectorCraft ``--control`` session."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .control import DEFAULT_CONTROL_PORT, DEFAULT_HOST, ControlConnection
from .errors import ControlError
from .find import ObjectQuery, bounds_xywh, coerce_query, filter_nodes


class LiveVectorCraft:
    """Talk to an already-running VectorCraft GUI via the JSON-lines control port."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_CONTROL_PORT, *, connect_timeout: float = 0.5, request_timeout: float = 60.0, connection: Optional[ControlConnection] = None, auto_reconnect: bool = True):
        self.host = host
        self.port = int(port)
        self.auto_reconnect = auto_reconnect
        self._duplicate_command: Optional[str] = None
        self._duplicate_command_resolved: bool = False
        if connection is not None:
            self._conn = connection
        else:
            self._conn = ControlConnection(host=host, port=port, connect_timeout=connect_timeout, request_timeout=request_timeout)


    @property
    def connection(self) -> ControlConnection:
        return self._conn


    @property
    def is_connected(self) -> bool:
        """True while the underlying control socket is open (``close()`` or the app's 5-minute idle timeout make it False)."""
        return self._conn.is_open


    def reconnect(self) -> "LiveVectorCraft":
        """Open a fresh control connection to the same host/port (the app closes idle connections after 5 minutes)."""
        self._conn.connect()
        self._duplicate_command = None
        self._duplicate_command_resolved = False
        return self


    def ensure_connected(self) -> "LiveVectorCraft":
        """Reconnect if the socket was closed; raises ``ControlError`` if ``auto_reconnect`` is off."""
        if not self.is_connected:
            if not self.auto_reconnect:
                raise ControlError("control connection is closed (auto_reconnect=False); call reconnect()")
            self.reconnect()
        return self


    def close(self) -> None:
        self._conn.close()


    def __enter__(self) -> "LiveVectorCraft":
        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


    def call(self, method: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        self.ensure_connected()
        return self._conn.call(method, params)


    def execute(self, command: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        self.ensure_connected()
        return self._conn.engine_execute(command, params)


    def ui_inspect(self) -> Any:
        self.ensure_connected()
        return self._conn.ui_inspect()


    def activate_document(self, index: int) -> Any:
        return self.execute("document.activate", {"index": int(index)})


    def active_document_info(self) -> Any:
        """``file.info`` with no params: title, units, artboards, object counts, …"""
        return self.execute("file.info", {})


    @staticmethod
    def _document_name_aliases(*names: Optional[str]) -> set:
        aliases = set()
        for name in names:
            if not name:
                continue
            text = str(name)
            aliases.add(text)
            path = Path(text)
            aliases.add(path.name)
            aliases.add(path.stem)
        ## END for name in names...

        return {a for a in aliases if a}


    def _document_matches_name(self, expected: str, *, title: str = '', path: str = '') -> bool:
        live_aliases = self._document_name_aliases(title, path)
        expected_aliases = self._document_name_aliases(expected)
        return not live_aliases.isdisjoint(expected_aliases)


    def _active_document_identity(self) -> Tuple[Dict[str, Any], str, str]:
        info = self.active_document_info()
        assert isinstance(info, dict), f"file.info did not return a dict: {info!r}"
        title: str = str(info.get('title') or '')
        tree = self.inspect_document(depth=0, childLimit=0)
        raw_path: str = str((tree.get('path') if isinstance(tree, dict) else '') or '')
        return info, title, raw_path


    def assert_active_document(self, expected: str) -> Dict[str, Any]:
        """Fail unless the active tab is ``expected`` (``file.info`` title, on-disk filename, or either stem)."""
        info, title, raw_path = self._active_document_identity()
        assert self._document_matches_name(expected, title=title, path=raw_path), (
            f"active document is {title!r} (path={Path(raw_path).name!r}), expected {expected!r} — activate the right tab first (live.list_documents())"
        )
        return info


    def select_document(self, name: str) -> Dict[str, Any]:
        """Activate the open tab matching ``name`` (title, filename, or stem). No-op if it is already active."""
        info, title, raw_path = self._active_document_identity()
        if self._document_matches_name(name, title=title, path=raw_path):
            return info

        documents = self.list_documents()
        matches = [d for d in documents if self._document_matches_name(name, title=str(d.get('title') or ''), path=str(d.get('path') or ''))]
        titles = [d.get('title') for d in documents]
        assert matches, f"no open document matching {name!r}; open tabs: {titles}"
        assert len(matches) == 1, f"multiple open documents match {name!r}: {[d.get('title') for d in matches]}"
        self.activate_document(int(matches[0]['index']))
        return self.assert_active_document(name)


    def _probe_document_at(self, index: int) -> Optional[Dict[str, Any]]:
        try:
            self.activate_document(index)
        except ControlError:
            return None
        info = self.active_document_info()
        doc: Dict[str, Any] = {"index": index}
        if isinstance(info, dict):
            doc.update(info)
        else:
            doc["info"] = info
        return doc


    def list_documents(self, *, max_documents: int = 64) -> List[Dict[str, Any]]:
        """Enumerate open document tabs by probing ``document.activate`` indices.

        Restores the originally active document (matched by ``title``) when finished.
        """
        original_title: Optional[str] = None
        try:
            before = self.active_document_info()
            if isinstance(before, dict):
                original_title = before.get("title")
        except ControlError:
            original_title = None

        documents: List[Dict[str, Any]] = []
        for index in range(0, max_documents):
            doc = self._probe_document_at(index)
            if doc is None:
                break
            documents.append(doc)
        ## END for index in range(0, max_documents)...

        if not documents:
            return []

        restore_index = 0
        if original_title is not None:
            for doc in documents:
                if doc.get("title") == original_title:
                    restore_index = int(doc["index"])
                    break
            ## END for doc in documents...

        try:
            self.activate_document(restore_index)
        except ControlError:
            pass
        return documents


    def inspect_document(self, **params: Any) -> Any:
        """Control method ``document.inspect`` (layer tree / scoped ``nodes``)."""
        return self.call("document.inspect", params or {})


    def list_nodes(self, *, kind: Optional[str] = None, depth: Optional[int] = None, child_limit: Optional[int] = None, limit: Optional[int] = None, offset: Optional[int] = None, ids: Optional[Sequence[Any]] = None) -> Any:
        """Enumerate nodes via ``document.inspect`` scope options; returns the raw inspect result."""
        params: Dict[str, Any] = {}
        if kind is not None:
            params["kind"] = kind
        if depth is not None:
            params["depth"] = depth
        if child_limit is not None:
            params["childLimit"] = child_limit
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if ids is not None:
            params["ids"] = list(ids)
        if not params:
            params["depth"] = 0
            params["childLimit"] = 50
        return self.inspect_document(**params)


    def get_node(self, node_id: Any, *, summary: bool = True, depth: Optional[int] = None, child_limit: Optional[int] = None) -> Any:
        params: Dict[str, Any] = {"id": node_id, "summary": summary}
        if depth is not None:
            params["depth"] = depth
        if child_limit is not None:
            params["childLimit"] = child_limit
        return self.execute("document.node", params)


    def find_objects(self, query: Union[ObjectQuery, Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Find objects in the active document matching ``ObjectQuery`` (name/kind/text/size/…)."""
        q = coerce_query(query)
        nodes: List[Dict[str, Any]] = []

        if q.has_native_find_filters():
            find_params = q.to_find_params()
            if not find_params:
                raise ControlError("document.find needs at least one of name, kind, text")
            result = self.execute("document.find", find_params)
            matches = []
            if isinstance(result, dict):
                matches = list(result.get("matches") or [])
            elif isinstance(result, list):
                matches = result
            ids = [m.get("id") for m in matches if isinstance(m, dict) and "id" in m]
            if ids and q.has_size_or_flag_filters():
                enriched = self.inspect_document(ids=ids)
                if isinstance(enriched, dict) and isinstance(enriched.get("nodes"), list):
                    nodes = [dict(n) for n in enriched["nodes"] if isinstance(n, dict)]
                else:
                    nodes = [dict(m) for m in matches if isinstance(m, dict)]
            else:
                nodes = [dict(m) for m in matches if isinstance(m, dict)]
                if ids and not nodes:
                    enriched = self.inspect_document(ids=ids)
                    if isinstance(enriched, dict) and isinstance(enriched.get("nodes"), list):
                        nodes = [dict(n) for n in enriched["nodes"] if isinstance(n, dict)]
        else:
            # Size / flag / regex only: page document.inspect by kind or all kinds via limit/offset.
            page_size = q.limit if (q.limit is not None and q.limit > 0) else 200
            offset = 0
            collected: List[Dict[str, Any]] = []
            while True:
                params: Dict[str, Any] = {"limit": page_size, "offset": offset}
                if q.kind is not None:
                    params["kind"] = q.kind
                else:
                    # Without kind, request a shallow tree and flatten — better: inspect with limit only needs kind/ids.
                    # Use a broad kind-less inspect with depth for skeleton is not a flat list; use limit without kind
                    # is documented as listing when kind/ids/limit/offset present — kind optional with limit.
                    pass
                page = self.inspect_document(**params)
                batch: List[Dict[str, Any]] = []
                if isinstance(page, dict):
                    if isinstance(page.get("nodes"), list):
                        batch = [dict(n) for n in page["nodes"] if isinstance(n, dict)]
                    elif isinstance(page.get("layers"), list):
                        batch = _flatten_layer_nodes(page["layers"])
                if not batch:
                    break
                collected.extend(batch)
                total = page.get("total") if isinstance(page, dict) else None
                offset += len(batch)
                if total is not None and offset >= int(total):
                    break
                if len(batch) < page_size:
                    break
                if q.limit is not None and len(collected) >= q.limit * 4:
                    # enough candidates for client-side filter
                    break
            ## END while True...

            nodes = collected

        return filter_nodes(nodes, q)


    def document_path(self) -> Path:
        """On-disk path of the active document (from ``document.inspect``'s ``path``)."""
        tree = self.inspect_document(depth=0, childLimit=0)
        doc_path: str = str(tree.get('path') or '')
        assert doc_path, f"active document {tree.get('title')!r} has never been saved (no path) — save it first, then back it up"
        return Path(doc_path)


    def wait_for_background(self, timeout_sec: float = 120.0, poll_sec: float = 0.25) -> None:
        """Block until the desktop app's Background Save/Export jobs (``ui.inspect`` → ``background``) are done."""
        t_start: float = time.monotonic()
        while True:
            ui = self.ui_inspect()
            jobs: List[Any] = list(ui.get('background') or []) if isinstance(ui, dict) else []
            if not jobs:
                return
            assert (time.monotonic() - t_start) < timeout_sec, f"background jobs still running after {timeout_sec}s: {jobs}"
            time.sleep(poll_sec)
        ## END while True...


    def save_copy(self, out_path: Union[str, Path], timeout_sec: float = 120.0) -> Dict[str, Any]:
        """``file.saveCopy {path}`` → the document keeps its own path/title/modified state. Format comes from ``out_path``'s extension."""
        dest: Path = Path(out_path)
        reply = self.execute('file.saveCopy', {'path': str(dest.resolve())})
        assert reply != 'cancelled', f"file.saveCopy was cancelled (a dialog was opened) for '{dest.as_posix()}' — reply={reply!r}"
        if isinstance(reply, dict) and reply.get('pending'):
            raise RuntimeError(f"file.saveCopy opened a {reply['pending']!r} dialog instead of writing '{dest.as_posix()}'; answer it in the app or change the backup format")
        if isinstance(reply, dict) and reply.get('background'):
            self.wait_for_background(timeout_sec=timeout_sec)
        assert dest.exists(), f"backup not written: '{dest.as_posix()}' (reply={reply!r})"
        if isinstance(reply, dict) and reply.get('warnings'):
            print(f"  warnings for '{dest.name}': {reply['warnings']}")
        return reply if isinstance(reply, dict) else {'result': reply}


    def backup_document(self, date_str: Optional[str] = None, backup_subfolder: str = '_BAK', also_native_copy: bool = True, expect_title: Optional[str] = None) -> Dict[str, Path]:
        """Back up the active document into ``<doc folder>/_BAK/<stem>_<date><ext>`` (creating ``_BAK`` if needed).

        ``also_native_copy=True`` additionally writes a lossless ``<stem>_<date>.vectorcraft`` next to it, since a PDF/SVG-format
        copy of a PDF/SVG-backed document re-exports through that format and can lose VectorCraft-only state.
        """
        if expect_title is not None:
            self.assert_active_document(expect_title)

        def _subfn_unique_backup_path(backup_dir: Path, stem: str, dated: str, suffix: str) -> Path:
            candidate: Path = backup_dir.joinpath(f"{stem}_{dated}{suffix}")
            i: int = 0
            while candidate.exists():
                i += 1
                candidate = backup_dir.joinpath(f"{stem}_{dated}_{i}{suffix}")
            ## END while candidate.exists()...

            return candidate

        dated: str = date_str if date_str is not None else datetime.now().strftime('%Y-%m-%d')
        doc_path: Path = self.document_path()
        assert doc_path.exists(), f"document path reported by VectorCraft does not exist on disk: '{doc_path.as_posix()}'"
        backup_dir: Path = doc_path.parent.joinpath(backup_subfolder)
        backup_dir.mkdir(parents=True, exist_ok=True)

        targets: List[Tuple[str, Path]] = [('same_format', _subfn_unique_backup_path(backup_dir, doc_path.stem, dated, doc_path.suffix))]
        if also_native_copy and (doc_path.suffix.lower() != '.vectorcraft'):
            targets.append(('vectorcraft', _subfn_unique_backup_path(backup_dir, doc_path.stem, dated, '.vectorcraft')))

        out: Dict[str, Path] = {}
        print(f"backing up {doc_path.name!r} → '{backup_dir.as_posix()}'")
        for label, out_path in targets:
            self.save_copy(out_path)
            print(f"  [{label}] {out_path.as_posix()}  ({out_path.stat().st_size / 1024:.1f} KiB)")
            out[label] = out_path
        ## END for label, out_path in targets...

        return out


    def find_by_name(self, name: str, exact: bool = True, limit: int = 100) -> List[Dict[str, Any]]:
        """``document.find`` by Layers-panel name (substring match server-side; exact filter client-side)."""
        result = self.execute('document.find', {'name': name, 'limit': limit})
        matches: List[Dict[str, Any]] = list(result.get('matches') or []) if isinstance(result, dict) else list(result or [])
        if exact:
            exact_matches = [m for m in matches if str(m.get('name') or '') == name]
            if exact_matches:
                return exact_matches
        return matches


    def node_bounds(self, node_id: Any) -> Dict[str, float]:
        node = self.get_node(node_id, summary=True, depth=0)
        xywh = bounds_xywh(node['bounds'])
        assert xywh is not None, f"node {node_id} has no bounds"
        return xywh


    def unique_name(self, base_name: str) -> str:
        """``base_name``, else ``base_name_1``, ``base_name_2``, … until no layer/object has that exact name."""
        candidate: str = base_name
        i: int = 0
        while self.find_by_name(candidate, exact=True):
            i += 1
            candidate = f"{base_name}_{i}"
        ## END while self.find_by_name(candidate, exact=True)...

        return candidate


    def duplicate(self, ids: Sequence[Any]) -> List[Any]:
        """Duplicate ``ids`` in place and return the new ids (whatever the reply doesn't name is read from the selection)."""
        def _subfn_resolve_duplicate_command() -> Optional[str]:
            catalogue = self.connection.engine_commands()
            catalogue_ids = {c.get('id') for c in catalogue if isinstance(c, dict)}
            for candidate in ('layer.duplicate', 'object.duplicate', 'edit.duplicate'):
                if candidate in catalogue_ids:
                    return candidate
            ## END for candidate in ('layer.duplicate', 'object.duplicate', 'edit.duplicate')...
            return None

        if not self._duplicate_command_resolved:
            self._duplicate_command = _subfn_resolve_duplicate_command()
            self._duplicate_command_resolved = True
        id_list: List[Any] = list(ids)
        if self._duplicate_command is not None:
            reply = self.execute(self._duplicate_command, {'ids': id_list})
        else:
            reply = self.execute('object.move', {'ids': id_list, 'dx': 0.0, 'dy': 0.0, 'copy': True})
        new_ids: List[Any] = []
        if isinstance(reply, dict):
            new_ids = list(reply.get('ids') or ([reply['id']] if 'id' in reply else []))
        if not new_ids:
            tree = self.inspect_document(depth=0, childLimit=0)
            new_ids = [s for s in (tree.get('selection') or []) if s not in id_list]
        assert new_ids, f"duplicate of ids={id_list} produced no new ids (reply={reply!r})"
        return new_ids


    def place_file(self, path: Union[str, Path], rect: Mapping[str, float], name: Optional[str] = None, crop: Optional[str] = 'crop', link: bool = False) -> Tuple[List[Any], Dict[str, float]]:
        """``file.place`` at ``rect`` (``[x, y, w, h]`` document points, artboard top-left), optionally rename, and read back the actual bounds.

        ``crop=None`` omits the param: an SVG is then one group fitted by its art bounds, not a Clip Group framed to the viewBox.
        """
        file_path: Path = Path(path)
        params: Dict[str, Any] = {'path': str(file_path.resolve()), 'rect': [rect['x'], rect['y'], rect['width'], rect['height']], 'link': link}
        if crop is not None:
            params['crop'] = crop
        reply = self.execute('file.place', params)
        placed_ids: List[Any] = []
        if isinstance(reply, dict):
            placed_ids = list(reply.get('ids') or ([reply['id']] if 'id' in reply else []))
        if not placed_ids:
            tree = self.inspect_document(depth=0, childLimit=0)
            placed_ids = list(tree.get('selection') or [])
        assert placed_ids, f"file.place returned no ids for '{file_path.as_posix()}' (reply={reply!r})"
        if name is not None:
            self.execute('object.setProps', {'ids': placed_ids, 'name': name})
        actual = self.node_bounds(placed_ids[0])
        return placed_ids, actual


    @contextmanager
    def undo_group(self, label: str) -> Iterator[None]:
        """One undo step wrapping the body (``edit.undoGroup.begin`` / ``.end``)."""
        try:
            self.execute('edit.undoGroup.begin', {'label': label})
        except ControlError:
            self.execute('edit.undoGroup.begin', {})
        try:
            yield
        finally:
            self.execute('edit.undoGroup.end', {})


def _flatten_layer_nodes(layers: Sequence[Any], *, path: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base_path = list(path or [])
    for node in layers:
        if not isinstance(node, dict):
            continue
        entry = dict(node)
        node_path = base_path + ([node["id"]] if "id" in node else [])
        if "path" not in entry and node_path:
            entry["path"] = node_path
        out.append(entry)
        children = node.get("children") or node.get("layers") or []
        if isinstance(children, list) and children:
            out.extend(_flatten_layer_nodes(children, path=node_path))
    ## END for node in layers...

    return out
