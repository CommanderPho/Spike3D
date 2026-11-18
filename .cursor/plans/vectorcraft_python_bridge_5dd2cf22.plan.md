---
name: VectorCraft Python Bridge
overview: Build a Windows-ready Python package under `EXTERNAL/VectorCraft_Scripting` that wraps the installed `vectorcraft-cli` Rust binary (headless `run` / `commands` / `convert` / `info`), with helpers that match the README figure-placement workflow.
todos:
  - id: pkg-scaffold
    content: Create EXTERNAL/VectorCraft_Scripting pyproject + src/vectorcraft_scripting package skeleton
    status: completed
  - id: cli-discovery
    content: Implement Windows-aware vectorcraft-cli discovery and subprocess runner
    status: completed
  - id: client-api
    content: Implement VectorCraftClient.run/commands/convert/info with JSON line parsing
    status: completed
  - id: place-helpers
    content: Add PlaceSpec, place_into_template, and optional Plotly SVG export helper
    status: completed
  - id: readme
    content: Expand README with install, PATH/env, and client-based usage matching the original sketch
    status: completed
  - id: smoke
    content: Smoke-test against C:\Program Files\VectorCraft\vectorcraft-cli.exe on this machine
    status: completed
isProject: false
---

# VectorCraft Python Scripting Bridge

## Context

- Working sketch lives in [`EXTERNAL/VectorCraft_Scripting/README.md`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\README.md) (only file there today).
- The “Rust API” is the shipped CLI at `C:\Program Files\VectorCraft\vectorcraft-cli.exe` (also `vectorcraft.exe` for live `--control`).
- Confirmed CLI surface:

```text
vectorcraft-cli run [--in FILE] [--cmd ID [--params JSON]]... [--export FILE]... [--scale N]
vectorcraft-cli commands | convert | info | mcp | bench | perf
```

- `file.place` params match the README (`path`, `rect`, `crop`, `link`, …).
- `PATH` currently does **not** include VectorCraft; the bridge must resolve the Windows install path.

## Chosen approach

**Headless CLI subprocess bridge first** (matches the README). No MCP stdio client and no live TCP control client in v1 — those can be added later if interactive sessions are needed. All figure standardization runs through one-shot `vectorcraft-cli run` processes, which is the reliable Windows batch path.

## Layout

```
EXTERNAL/VectorCraft_Scripting/
  README.md                 # expand: install, PATH, API examples
  pyproject.toml            # small local package: vectorcraft_scripting
  src/vectorcraft_scripting/
    __init__.py             # public exports
    cli.py                  # binary discovery + subprocess runner
    client.py               # VectorCraftClient (run/commands/convert/info)
    place.py                # PlaceSpec + place_file helper
    figures.py              # plotly/SVG → place → export convenience
    errors.py               # VectorCraftError / CliNotFoundError
```

Not wired into Spike3D’s root `pyproject.toml` unless you ask later; install with `uv sync` / `uv pip install -e EXTERNAL/VectorCraft_Scripting` from that folder.

## Implementation details

### 1. Binary discovery (`cli.py`)

Resolve executable in order:

1. Explicit constructor/`VECTORCRAFT_CLI` env
2. `shutil.which("vectorcraft-cli")` / `vectorcraft-cli.exe`
3. Windows defaults: `C:\Program Files\VectorCraft\vectorcraft-cli.exe` (and `(x86)` if needed)

Raise `CliNotFoundError` with those searched paths if missing. Always pass absolute paths into `--in` / `--export` / `file.place` params (Windows-safe).

### 2. Core client (`client.py`)

```python
class VectorCraftClient:
    def run(self, *, input: Path | None = None, commands: Sequence[tuple[str, dict] | str] = (), exports: Sequence[Path] = (), scale: float = 1.0) -> list[dict]: ...
    def commands(self) -> list[dict]: ...
    def convert(self, src: Path, dst: Path, **opts) -> dict: ...
    def info(self, path: Path) -> dict: ...
```

- Build argv exactly as the README: `--cmd id --params '{...}'` pairs; `--params` only when non-empty.
- Parse stdout as **one JSON value per line** (CLI prints one JSON result per step); raise on non-zero exit with stderr.
- Use `subprocess.run(..., check=False, capture_output=True, text=True)` — no shell, so spaces in `Program Files` are fine.

### 3. High-level place helpers (`place.py`, `figures.py`)

- `PlaceSpec(path, rect=None, at=None, crop="crop", link=False, **extra)` → params dict for `file.place`.
- `place_into_template(template, places, exports, ...)` → one `run()` with multiple `--cmd file.place` then exports.
- `export_plotly_figure(fig, chart_svg, template, rect, exports)` — README workflow: Kaleido write SVG → place → export (optional dependency; import kaleido/plotly only inside that helper).

### 4. README

Document:

- Windows install location and optional PATH / `VECTORCRAFT_CLI`
- Install/use of the Python package
- The README Plotly example rewritten against `VectorCraftClient` / `export_plotly_figure`
- Note that `vectorcraft-cli` must be installed (already present on this machine)

## Out of scope (v1)

- MCP stdio client
- Live `vectorcraft --control PORT` TCP JSON-lines client
- Shipping a `.vectorcraft` template binary in-repo
- Integrating into `pyphocorehelpers` / Spike3D deps

## Smoke check (after implement)

Run against the installed binary:

```powershell
uv run python -c "from vectorcraft_scripting import VectorCraftClient; print(VectorCraftClient().commands()[0]['id'])"
```

Optionally a tiny `run` with `file.new` + `--export` to a temp `.svg` under the scripting folder.