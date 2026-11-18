---
name: Fix Vispy Example Env Launch
overview: Ensure examples are launched with the exact same Python interpreter and runtime environment context as the browser process, and add lightweight diagnostics to verify which interpreter/env is actually used.
todos: []
isProject: false
---

# Fix Example Launch Environment in Vispy Browser

## Goal

Make `Run Example` reliably execute with the same interpreter/environment as the running browser process and make that visible for quick verification.

## Findings

- In `[h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py)`, `run_example` already uses `sys.executable`, but launch context is implicit.
- In `[h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)`, script execution is via `runpy.run_path(...)`, but there is no explicit startup diagnostic showing interpreter/environment in the new console.

## Implementation Plan

1. Update process launch in `run_example` to explicitly preserve environment context:
  - Build command with resolved interpreter path: `python_exe = str(Path(sys.executable).resolve())`.
  - Pass explicit inherited env: `env=os.environ.copy()`.
  - Set `cwd=str(path.parent)` so imports and resource lookups run from the example’s folder context.
  - Keep platform-specific `creationflags` behavior as-is on Windows.
2. Add startup diagnostics in `_run_vispy_example.py`:
  - Print (or otherwise emit) `sys.executable` and key env indicators (`VIRTUAL_ENV`, `CONDA_PREFIX`, optionally `PYTHONPATH`) at script start.
  - Keep diagnostics minimal and before `runpy.run_path(...)` so they always appear in the spawned console.
3. Harden argument handling slightly (minimal change):
  - Ensure the wrapper resolves and validates `script_path` before execution.
  - If invalid, fail with a clear message in the console instead of silent exit.
4. Verify behavior:
  - Launch one example and confirm console shows expected interpreter path and env marker.
  - Confirm no regressions in window title patching logic.
  - Run lints on edited files and clear any new warnings.

## Files To Change

- `[h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py)`
- `[h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\_run_vispy_example.py)`

