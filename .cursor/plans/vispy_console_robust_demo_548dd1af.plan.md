---
name: Vispy console robust demo
overview: Wire the Vispy example browser’s embedded `PhoCodeConsoleWidget` with an explicit initial namespace, documentation of `%run` / `!` / Python calls, and a small refactor so the same async `QProcess` launcher used by the Run button can be invoked from the console without blocking the UI (unlike `%run` or `!`).
todos:
  - id: refactor-qprocess
    content: Extract _start_example_process(name, path) from run_example; run_example calls it
    status: completed
  - id: console-namespace
    content: Pass namespace= to PhoCodeConsoleWidget with browser, paths, vispy_python; add run_vispy_example(name) if desired
    status: completed
  - id: welcome-text
    content: Replace console initial text with async vs %run vs ! guidance and example commands
    status: completed
  - id: wrapper-docstring
    content: Extend _run_vispy_example.py module docstring for argv and console integration
    status: completed
isProject: false
---

# Robust PhoCodeConsoleWidget integration for Vispy examples

## Current behavior

- `[vispy_EXAMPLES.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py)` already embeds `[PhoCodeConsoleWidget](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PhoCodeConsoleWidget.py)` as `self._run_console` and streams **async** `QProcess` stdout into it when **Run Example** is clicked (see `run_example`, `_on_example_process_ready_read`, merged channels).
- In-process script execution (`%run` / `run`) and shell lines (`!`) in `PhoCodeConsoleWidget` use `runpy.run_path` and `subprocess.run` respectively — both **block the Qt main thread** until the script/command finishes (see `[Console.runCmd](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqtgraph\console\Console.py)` forwarding to `execSingle`, and `_run_shell_line` in PhoCodeConsoleWidget). That is acceptable for quick commands but **not** equivalent to the Run button for long-running vispy windows.

## Design principle for “robust” use


| Mechanism                                    | Process model    | Blocks browser UI?                  | Matches `_run_vispy_example.py` title/env patching?                 |
| -------------------------------------------- | ---------------- | ----------------------------------- | ------------------------------------------------------------------- |
| Run button / shared `QProcess` path          | Separate process | No (async handlers)                 | Yes                                                                 |
| `browser.run_example()` (same internal path) | Separate process | No                                  | Yes                                                                 |
| `!python ... _run_vispy_example.py ...`      | Separate process | **Yes** (waits in `subprocess.run`) | Yes                                                                 |
| `%run "_run_vispy_example.py" ...`           | In-process       | **Yes**                             | Yes, but same interpreter + global Canvas monkey-patch side effects |


The demonstration should make this table obvious in the UI text and steer users toward **async launcher** for normal example runs.

## Implementation (all in Spike3D `LibrariesExamples/vispy/`)

### 1. Initial namespace for the console

In `VispyExampleBrowser.create_ui`, construct `PhoCodeConsoleWidget` with a `namespace=` dict (instead of default empty dict) populated after `self.examples_dir` / `self.examples` exist:

- `browser` → `self` so users can run `browser.run_example()` (same as the button: requires a list selection; **non-blocking**).
- `vispy_examples_dir` → `self.examples_dir`.
- `vispy_run_helper` → resolved `[Path](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)` to the wrapper script.
- `vispy_python` → `_subprocess_python_executable()` (same interpreter rule as `run_example`).
- Optional: `run_vispy_example(name)` → small method on the browser (or a module-level closure) that resolves `name` against `self.examples`, then starts the same QProcess used by `run_example` — lets the console run **by example id** without touching the list, still **non-blocking**.

Use `enable_shell_commands=True` explicitly in the constructor for clarity (current default is already True).

### 2. Refactor `run_example` to call one internal starter (minimal diff)

Extract the “build python + wrapper + args + env + connect signals + start” block from `run_example` into a private method like `_start_example_process(self, name, path)` that returns `True`/`False` or raises only on programmer error. `run_example` keeps UI selection logic and calls `_start_example_process`. `run_vispy_example(name)` reuses `_start_example_process` after resolving `(name, path)` from `self.examples`.

This keeps one code path for wrapper invocation and avoids drift.

### 3. Rich initial `text=` on the console widget

Replace the single-sentence intro with a short, plain-Python-comment-friendly blurb (no doctest `>>>`), including:

- **Preferred:** select example → `browser.run_example()` or `run_vispy_example("relative/name")` (once added).
- **Run button** is the same subprocess path.
- `**%run`** / `**run`:** in-process via `runpy`; show one example line with quoted paths for `[_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)`; note **UI blocks** until finish and shared-process caveats for vispy/Qt.
- `**!`:** shell subprocess; note **UI blocks** until the child exits; Windows quoting reminder; mention `enable_shell_commands=False` for locked-down hosts (aligns with PhoCodeConsoleWidget docstring).

Use real paths in the welcome text via `Path(__file__).resolve().parent` when building the string so copy-paste is locally correct.

### 4. Document `_run_vispy_example.py` for console users

Extend the module docstring at the top of `[_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)` with 2–3 lines: intended argv shape (`<example_name> <script_path> [extra args]`), and that the Vispy browser invokes it via `QProcess`; from `PhoCodeConsoleWidget`, prefer the browser’s async helpers over `%run`/ `!` for interactive responsiveness.

## Files touched

- `[LibrariesExamples/vispy/vispy_EXAMPLES.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py)` — namespace, welcome text, optional `run_vispy_example`, `_start_example_process` refactor.
- `[LibrariesExamples/vispy/_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)` — docstring only.

No change required to `[PhoCodeConsoleWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PhoCodeConsoleWidget.py)` unless you later want a **non-blocking** `!` variant (out of scope here; would need API/design discussion).

## Verification

- Launch `python vispy_EXAMPLES.py`, select an example, type `browser.run_example()` in the console → same behavior as the button (log + non-blocking).
- If `run_vispy_example("...")` is added, call it with a known scanned name → same.
- Optional: `%run`/`!` smoke with a trivial script to confirm console still works (understanding UI blocks for duration).

