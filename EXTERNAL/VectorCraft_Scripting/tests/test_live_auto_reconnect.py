"""Unit tests for LiveVectorCraft idle/stale-connection auto-reconnect."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import pytest

from vectorcraft_scripting.errors import ControlError
from vectorcraft_scripting.live import LiveVectorCraft


class FakeControlConnection:
    """Minimal ControlConnection stand-in for reconnect tests."""

    def __init__(self, *, fail_first_call: bool = False, app_error: bool = False):
        self._open = True
        self.fail_first_call = fail_first_call
        self.app_error = app_error
        self.call_count = 0
        self.connect_count = 0
        self.last_method: Optional[str] = None
        self.last_params: Optional[Mapping[str, Any]] = None


    @property
    def is_open(self) -> bool:
        return self._open


    def connect(self) -> None:
        self.connect_count += 1
        self._open = True


    def close(self) -> None:
        self._open = False


    def call(self, method: str, params: Optional[Mapping[str, Any]] = None, *, id: Optional[int] = None) -> Any:
        self.call_count += 1
        self.last_method = method
        self.last_params = params
        if self.app_error:
            raise ControlError("`ui.inspect` failed", method=method, reply={"ok": False, "error": "`ui.inspect` failed"})
        if self.fail_first_call:
            self.fail_first_call = False
            self.close()
            cause = ConnectionAbortedError(10053, "An established connection was aborted by the software in your host machine")
            raise ControlError(f"control connection failed during `{method}`: {cause}", method=method) from cause
        return {"ok": True, "method": method, "params": dict(params) if params else {}}


    def engine_execute(self, command: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        body: Dict[str, Any] = {"command": command}
        if params is not None:
            body["params"] = dict(params)
        return self.call("engine.execute", body)


    def ui_inspect(self) -> Any:
        return self.call("ui.inspect", {})


    def engine_commands(self) -> Any:
        return self.call("engine.commands", {})


def test_control_error_is_connection_lost_from_oserror_cause():
    cause = ConnectionAbortedError(10053, "aborted")
    exc = ControlError("control connection failed during `engine.execute`: aborted", method="engine.execute")
    exc.__cause__ = cause
    assert exc.is_connection_lost is True


def test_control_error_is_connection_lost_from_message_prefix():
    exc = ControlError("control connection closed while awaiting `ui.inspect`", method="ui.inspect")
    assert exc.is_connection_lost is True


def test_control_error_app_level_is_not_connection_lost():
    exc = ControlError("unknown command", method="engine.execute", reply={"ok": False, "error": "unknown command"})
    assert exc.is_connection_lost is False


def test_auto_reconnect_retries_once_after_transport_failure():
    fake = FakeControlConnection(fail_first_call=True)
    live = LiveVectorCraft(connection=fake, auto_reconnect=True)

    result = live.call("ui.inspect", {})

    assert result["method"] == "ui.inspect"
    assert fake.call_count == 2
    assert fake.connect_count == 1
    assert live.is_connected is True


def test_auto_reconnect_retries_execute_after_idle_abort():
    fake = FakeControlConnection(fail_first_call=True)
    live = LiveVectorCraft(connection=fake, auto_reconnect=True)

    result = live.execute("document.open", {"path": "x.vectorcraft"})

    assert result["method"] == "engine.execute"
    assert fake.call_count == 2
    assert fake.connect_count == 1


def test_auto_reconnect_false_does_not_retry():
    fake = FakeControlConnection(fail_first_call=True)
    live = LiveVectorCraft(connection=fake, auto_reconnect=False)

    with pytest.raises(ControlError) as caught:
        live.call("ui.inspect", {})

    assert caught.value.is_connection_lost is True
    assert fake.call_count == 1
    assert fake.connect_count == 0


def test_app_level_control_error_is_not_retried():
    fake = FakeControlConnection(app_error=True)
    live = LiveVectorCraft(connection=fake, auto_reconnect=True)

    with pytest.raises(ControlError) as caught:
        live.call("ui.inspect", {})

    assert caught.value.is_connection_lost is False
    assert fake.call_count == 1
    assert fake.connect_count == 0


def test_ensure_connected_reconnects_when_socket_already_closed():
    fake = FakeControlConnection()
    fake.close()
    live = LiveVectorCraft(connection=fake, auto_reconnect=True)

    live.ensure_connected()

    assert fake.connect_count == 1
    assert live.is_connected is True
