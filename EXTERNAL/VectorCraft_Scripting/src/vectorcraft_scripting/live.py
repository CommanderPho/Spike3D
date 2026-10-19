"""High-level live client for a running VectorCraft ``--control`` session."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .control import DEFAULT_CONTROL_PORT, DEFAULT_HOST, ControlConnection
from .errors import ControlError
from .find import ObjectQuery, coerce_query, filter_nodes


class LiveVectorCraft:
    """Talk to an already-running VectorCraft GUI via the JSON-lines control port."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_CONTROL_PORT, *, connect_timeout: float = 0.5, request_timeout: float = 60.0, connection: Optional[ControlConnection] = None):
        self.host = host
        self.port = int(port)
        if connection is not None:
            self._conn = connection
        else:
            self._conn = ControlConnection(host=host, port=port, connect_timeout=connect_timeout, request_timeout=request_timeout)


    @property
    def connection(self) -> ControlConnection:
        return self._conn


    def close(self) -> None:
        self._conn.close()


    def __enter__(self) -> "LiveVectorCraft":
        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


    def call(self, method: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        return self._conn.call(method, params)


    def execute(self, command: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        return self._conn.engine_execute(command, params)


    def ui_inspect(self) -> Any:
        return self._conn.ui_inspect()


    def activate_document(self, index: int) -> Any:
        return self.execute("document.activate", {"index": int(index)})


    def active_document_info(self) -> Any:
        """``file.info`` with no params: title, units, artboards, object counts, …"""
        return self.execute("file.info", {})


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
