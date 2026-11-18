---
name: PhoCodeConsole script run
overview: Extend `PhoCodeConsoleWidget` to run `.py` files with real `__main__` / `__file__` / `sys.argv` semantics via `runpy.run_path`, merge results into the interactive namespace, and support optional `!` shell passthrough (configurable). Wire the pipeline UI to the subclass so the docked console actually uses it.
todos:
  - id: impl-phocodeconsole
    content: "Implement execSingle override: %run/run (runpy + argv + namespace merge), ! subprocess gated by enable_shell_commands"
    status: completed
  - id: wire-mainpipeline-ui
    content: Point MainPipelineWindow.ui custom widget to PhoCodeConsoleWidget; fix uic registration if loadUi fails
    status: completed
  - id: smoke-test
    content: "Manual smoke: __main__ block, argv, merged names, shell output, disabled shell flag"
    status: completed
isProject: false
---

# Terminal-like script execution for PhoCodeConsoleWidget

## Context

The parent class is the vendored `[Console.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqtgraph\console\Console.py)`. It executes input with `eval` / `exec` only: no `__file__`, no `__name__ == "__main__"` block behavior, and no `sys.argv` for a script—so `if __name__ == "__main__":` in a file never runs the way users expect from a terminal. Multiline REPL continuation is driven by a brittle `'unexpected EOF'` string match (unchanged here unless you want a follow-up using `code.compile_command`).

`[PhoCodeConsoleWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\PhoCodeConsoleWidget.py)` is currently an empty subclass; the docked console in `[MainPipelineWindow.ui](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Windows\MainPipelineWindow.ui)` still promotes base `ConsoleWidget`, so new behavior must be implemented in `PhoCodeConsoleWidget` **and** the UI promotion updated (or the console will never see it).

## Behavior to add

1. **Run Python file (terminal-like)**
  - Recognize a line-oriented magic (proposed: `%run <path> [args...]` and an alias `run <path> [args...]` when the line starts with that token), using `shlex.split` so quoted paths work on Windows.  
  - Resolve paths with `pathlib.Path`: if not absolute, resolve against `os.getcwd()` (document that changing cwd in the app affects resolution).  
  - Execute with `[runpy.run_path](https://docs.python.org/3/library/runpy.html#runpy.run_path)`: `run_name="__main__"`, `init_globals` seeded from the current console globals (`self.globals()`), temporary `sys.argv = [str(path), *args]`, then restore.  
  - Merge the returned namespace into `self.localNamespace`, excluding `__builtins__` and `__console__` (and other unsafe overwrites as needed) so definitions and top-level assignments remain visible in the REPL.  
  - On failure (`FileNotFoundError`, `SyntaxError`, etc.), route through the same `write` / `displayException` pattern as the parent so stack traces appear in the widget.
2. **Shell passthrough (`!`)** — per your choice
  - If `enable_shell_commands` is True (default), a line starting with `!` runs the rest as a subprocess; stream or capture stdout/stderr and append to the console via `write`.  
  - On Windows, use `shell=True` for `!` so builtins like `dir` work; document the security implication (user-controlled command).  
  - Constructor flag: `enable_shell_commands: bool = True` so hosts can disable `!` in locked-down contexts.
3. **Implementation shape**
  - Override `execSingle` only (magics are single-shot lines; multiline continuation stays on `super().execMulti`).  
  - Early-return branches: strip the line, detect `%run` / `run` / `!`, else `super().execSingle(cmd)`.  
  - Keep imports minimal: `runpy`, `os`, `shlex`, `subprocess`, `pathlib.Path`, `sys`.  
  - Match existing style: single-line signatures and calls where possible; two blank lines between methods on the subclass.

## UI integration (required for real use)

Update `[MainPipelineWindow.ui](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Windows\MainPipelineWindow.ui)`:

- Change the console `<widget class="...">` and `<customwidget> <class>` / `<header>` to `PhoCodeConsoleWidget` and `pyphoplacecellanalysis.GUI.Qt.Widgets.PhoCodeConsoleWidget`.

If any loader path fails at runtime, register the custom widget in the same place `uic.loadUi` is called (likely the main window module)—only add that if testing shows it is necessary.

## Non-goals (explicit)

- **Non-blocking execution**: long scripts still run on the Qt GUI thread (same as today); fixing that would be a separate QThread / worker design.  
- **Patching vendored** `Console.py`: avoid unless you later want `compile_command`-based multiline for everyone.

## Quick verification

- From the docked console: `%run path\to\snippets\hello.py` (with a small script using `if __name__ == "__main__": print(sys.argv)`).  
- `!echo test` (Windows) shows output; with `enable_shell_commands=False`, `!` falls through or is rejected with a short message.

