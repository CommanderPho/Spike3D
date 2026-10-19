"""JSON-lines TCP client for VectorCraft's live ``--control`` protocol."""

from __future__ import annotations

import json
import socket
from typing import Any, Dict, Mapping, Optional

from .errors import ControlError

DEFAULT_HOST = "127.0.0.1"
DEFAULT_CONTROL_PORT = 7979


class ControlConnection:
    """One persistent loopback connection: one JSON request line → one reply line."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_CONTROL_PORT, *, connect_timeout: float = 0.5, request_timeout: float = 60.0):
        self.host = host
        self.port = int(port)
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._sock: Optional[socket.socket] = None
        self._reader = None
        self._next_id = 1
        self.connect()


    def connect(self) -> None:
        self.close()
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        sock.settimeout(self.request_timeout)
        self._sock = sock
        self._reader = sock.makefile("r", encoding="utf-8", newline="\n")


    def close(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


    def __enter__(self) -> "ControlConnection":
        return self


    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


    def _alloc_id(self) -> int:
        req_id = self._next_id
        self._next_id += 1
        return req_id


    def call(self, method: str, params: Optional[Mapping[str, Any]] = None, *, id: Optional[int] = None) -> Any:
        """Send ``{"method", "params", "id"}`` and return ``result``, or raise ``ControlError``."""
        if self._sock is None or self._reader is None:
            raise ControlError("control connection is closed", method=method)
        req_id = self._alloc_id() if id is None else id
        request: Dict[str, Any] = {"id": req_id, "method": method}
        if params is not None:
            request["params"] = dict(params)
        payload = json.dumps(request, separators=(",", ":")) + "\n"
        try:
            self._sock.sendall(payload.encode("utf-8"))
            line = self._reader.readline()
        except OSError as e:
            self.close()
            raise ControlError(f"control connection failed during `{method}`: {e}", method=method) from e
        if not line:
            self.close()
            raise ControlError(f"control connection closed while awaiting `{method}`", method=method)
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as e:
            self.close()
            raise ControlError(f"invalid JSON reply for `{method}`: {line[:200]!r}", method=method) from e
        if not isinstance(reply, dict):
            raise ControlError(f"expected object reply for `{method}`, got {type(reply).__name__}", method=method, reply={"raw": reply})
        if reply.get("ok") is False:
            raise ControlError(str(reply.get("error") or f"`{method}` failed"), method=method, reply=reply)
        if "result" not in reply and reply.get("ok") is not True:
            # Some replies may be ok:true with result; treat missing ok+error as soft success if result present.
            if "error" in reply:
                raise ControlError(str(reply.get("error")), method=method, reply=reply)
        return reply.get("result")


    def ui_inspect(self) -> Any:
        return self.call("ui.inspect", {})


    def engine_execute(self, command: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        body: Dict[str, Any] = {"command": command}
        if params is not None:
            body["params"] = dict(params)
        return self.call("engine.execute", body)


    def engine_commands(self) -> Any:
        return self.call("engine.commands", {})


def probe_control(host: str = DEFAULT_HOST, port: int = DEFAULT_CONTROL_PORT, *, connect_timeout: float = 0.35) -> Optional[Any]:
    """Return ``ui.inspect`` result if ``host:port`` speaks the control protocol, else ``None``."""
    try:
        with ControlConnection(host=host, port=port, connect_timeout=connect_timeout) as conn:
            return conn.ui_inspect()
    except (ControlError, OSError, TimeoutError):
        return None
