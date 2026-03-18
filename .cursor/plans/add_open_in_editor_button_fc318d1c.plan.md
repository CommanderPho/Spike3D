---
name: Add Open in Editor Button
overview: Add a button below "Run Example" in the Vispy Examples Browser that opens the selected example file in the system default application (editor for .py files), using minimal edits to vispy_EXAMPLES.py only.
todos: []
isProject: false
---

# Add "Open in Default Editor" Button to Vispy Examples Browser

## Context

- **[vispy_EXAMPLES.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\vispy_EXAMPLES.py)** builds the UI: left panel has the example list, then "Run Example", then stretch. The selected example is resolved to `(name, path, description)` with `path` a `Path` to the example `.py` file.
- **[_run_vispy_example.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy_run_vispy_example.py)** is only the subprocess-runner script; no UI changes needed there.

## Implementation (vispy_EXAMPLES.py only)

1. **New button in the left panel**
  In `create_ui`, add a second button directly below the existing "Run Example" button (before `left_layout.addStretch()`), e.g. label **"Open in Default Editor"**. Store it as `self.open_editor_button`, set `setEnabled(False)` initially, and connect `clicked` to a new slot (e.g. `open_in_editor`).
2. **Open-in-editor handler**
  Implement `open_in_editor(self)` that:
  - Gets the current list item and resolves the example the same way as `run_example` (using `_canonical_name` and the `self.examples` list to get `path`).
  - If no selection or no path, return.
  - Opens the file with the system default application:
    - **Windows**: `os.startfile(str(path))`
    - **macOS**: `subprocess.Popen(["open", str(path)])`
    - **Linux**: `subprocess.Popen(["xdg-open", str(path)])`
  - Optionally wrap in try/except and show a `QMessageBox` on failure (e.g. "Could not open file in default editor").
3. **Enable/disable with selection**
  In `on_example_selected`, after enabling/disabling `self.run_button`, do the same for `self.open_editor_button` (enabled when an example is selected and the file is loadable; i.e. when `self.run_button` is enabled).

## Code touch points

- **create_ui** (around lines 186–191): add the new button and its `clicked` connection after the Run Example button.
- **on_example_selected** (around 251–272): set `self.open_editor_button.setEnabled(True)` or `False` in the same branches where `self.run_button.setEnabled` is set.
- **New method** `open_in_editor`: add after `run_example` (after line 296), following the existing two-blank-lines style.

No changes to `_run_vispy_example.py`; it is not involved in opening the file in the editor.