---
name: Stale control reconnect
overview: Fix first-call failures after idle by retrying control RPCs once after transport-level `ControlError`, matching the existing `auto_reconnect=True` intent (today reconnect only runs when the socket is already marked closed).
todos:
  - id: classify-transport-errors
    content: Add ControlError connection-lost classifier (property or helper) for transport-only failures
    status: completed
  - id: with-auto-reconnect
    content: Implement LiveVectorCraft._with_auto_reconnect and wire call/execute/ui_inspect + engine.commands path
    status: completed
  - id: docs-readme
    content: Update live.py docstrings and README live section for idle retry behavior
    status: completed
  - id: unit-tests
    content: "Add pytest with fake connection: fail once on transport, succeed after reconnect; no retry on app-level ControlError"
    status: completed
isProject: false
---

# Stale live control connection auto-retry

## Diagnosis

```mermaid
sequenceDiagram
    participant NB as Notebook
    participant Live as LiveVectorCraft
    participant VC as VectorCraft_control_port

    Note over NB,VC: Idle 5+ min; server closes TCP
    NB->>Live: execute(...)
    Live->>Live: ensure_connected()
    Note over Live: is_open True (_sock still set)
    Live->>VC: send JSON line
    VC--xLive: recv ConnectionAbortedError 10053
    Live->>Live: close socket; raise ControlError
    NB->>Live: execute(...) again
    Live->>Live: ensure_connected() sees closed
    Live->>VC: new TCP connection
    VC-->>Live: OK
```

Relevant code today:

- [`control.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\control.py): `call()` closes the socket and wraps `OSError` (including `ConnectionAbortedError`) in `ControlError` on **read** (and send).
- [`live.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\live.py): `ensure_connected()` only reconnects when `is_connected` is false — but [`is_open`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\control.py) is true whenever `_sock` / `_reader` exist, **not** when the peer still accepts traffic.

Docstrings already mention the app’s 5-minute idle close; behavior just does not match user expectation of `auto_reconnect=True`.

## Recommended fix (client-side, minimal)

Centralize transport recovery in `LiveVectorCraft` so notebook users need not re-run cells.

### 1. Classify reconnectable `ControlError`

Add a small helper (in [`errors.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\errors.py) as `ControlError.is_connection_lost` property, or a module-level `_is_connection_lost(exc)` in `live.py`) that returns **True** only for transport breaks, e.g.:

- `exc.__cause__` is `OSError` (covers WinError 10053 / reset / broken pipe)
- Message prefixes already emitted by `ControlConnection.call`: `"control connection failed during"`, `"control connection closed while awaiting"`, `"invalid JSON reply for"` (these paths call `close()`)

Return **False** for application/protocol failures (`ok: false` from VectorCraft, bad reply shape without closing, etc.) so we do not double-run successful-but-rejected commands.

### 2. `_with_auto_reconnect` wrapper on `LiveVectorCraft`

In [`live.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\live.py):

```python
def _with_auto_reconnect(self, fn: Callable[[], Any]) -> Any:
    self.ensure_connected()
    try:
        return fn()
    except ControlError as exc:
        if not self.auto_reconnect or not _is_connection_lost(exc):
            raise
        self.reconnect()
        return fn()
```

Route all control entry points through it:

| Method | Change |
|--------|--------|
| `call` | `return self._with_auto_reconnect(lambda: self._conn.call(...))` |
| `execute` | same via `engine_execute` |
| `ui_inspect` | same via `ui_inspect` |
| `duplicate` → `engine_commands` | replace `self.connection.engine_commands()` with `self.call("engine.commands", {})` so idle stale sockets are covered |

Remove redundant `ensure_connected()` from those three public methods (wrapper owns it).

### 3. Docstring / README touch-up

- Update `ensure_connected` / `auto_reconnect` docs: reconnect runs when the socket is closed **or** after one transport failure on the next RPC (idle server timeout).
- Add one sentence under [README live section](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\README.md) (~line 115): idle connections are retried automatically when `auto_reconnect=True` (default).

No notebook edits required.

## Testing (lightweight, no VectorCraft GUI)

There is no `tests/` tree today. Add a focused unit test file under `EXTERNAL/VectorCraft_Scripting/tests/` using a fake `ControlConnection` (or mock socket) injected via `LiveVectorCraft(..., connection=fake)`:

1. First `call` raises `ControlError("control connection failed during `ui.inspect`: ...", method=...)` with `__cause__=ConnectionAbortedError(...)`, sets fake `is_open` false after raise (mirror real `ControlConnection`).
2. Assert `_with_auto_reconnect` / `live.call` invokes `connect()` once and succeeds on second attempt.

Optional: test that a server-side `ControlError` with `reply={"ok": False}` is **not** retried (mock returns same error twice; assert only one underlying call).

Run with `uv run pytest` from `EXTERNAL/VectorCraft_Scripting`.

## Out of scope (unless you want them later)

- **Proactive idle ping** before every command (`ui.inspect`) — adds latency; retry-on-failure is enough for the reported bug.
- **Changing VectorCraft server idle timeout** — would require upstream app changes.
- **TCP keepalive tuning** on `connect()` — may help detection timing but does not replace retry logic.

## Risk note

A single transparent retry can theoretically repeat a command if the server processed the request but the reply was lost. That is rare on loopback JSON-lines RPC; the alternative (surfacing the error) is what you hit today. Document that `auto_reconnect=False` preserves strict fail-fast behavior.
