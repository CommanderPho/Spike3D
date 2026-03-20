---
name: Camera view menu bar
overview: Add a View (or Camera) menu on the dock window’s QMenuBar with four actions that set `scene.TurntableCamera` elevation/azimuth presets and refresh the canvas, implemented entirely in `Volumentric2DTimeSeriesPlotter.buildUI` plus small helper methods in [predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py).
todos:
  - id: imports-presets
    content: Add QtGui/QAction import and shared TurntableCamera default + preset angle constants
    status: completed
  - id: helper-apply
    content: Add _apply_turntable_camera_preset(elevation, azimuth) with isinstance guard and canvas.update()
    status: completed
  - id: buildui-menu
    content: In buildUI, attach View menu with Top/Left/Right/Perspective QAction -> helper; align initial camera with Perspective constants
    status: completed
isProject: false
---

# Camera presets via menu bar (Volumentric2DTimeSeriesPlotter)

## Context

- The viewer window is `[PhoDockAreaContainingWindow](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/DockAreaWrapper.py)` (`QMainWindow`), stored as `self.main_window` in `[buildUI](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)` (~1823–1914).
- The 3D view uses `scene.TurntableCamera(fov=45.0, elevation=30.0, azimuth=135.0)` (~1828). VisPy’s `TurntableCamera` exposes `elevation`, `azimuth`, and `roll` (default camera `up` is `+z` per VisPy `BaseCamera`).

## UI choice

- Add a `**QMenuBar` menu** (e.g. **View** or **Camera**) on `root_dockAreaWindow.menuBar()` with four `**QAction`** entries: **Top**, **Left**, **Right**, **Perspective**.
- This matches “menubar at the top” and is consistent with Qt; menu entries are the usual pattern (if you later want literal toolbar buttons, the same `QAction` objects can be added to a `QToolBar` with `addAction`).

## Camera angles (initial values)

Use `**scene.TurntableCamera`** semantics (VisPy docs: at `elevation=0`, `azimuth=0`, camera looks along `+y`; `elevation` rotates around scene `x`, `azimuth` around scene `z`):


| Preset          | elevation | azimuth | Notes                                                                |
| --------------- | --------- | ------- | -------------------------------------------------------------------- |
| **Perspective** | `30.0`    | `135.0` | Match current constructor so “Perspective” restores the default view |
| **Top**         | `90.0`    | `0.0`   | Look down onto the XY plane (arena) from `+Z`                        |
| **Left**        | `0.0`     | `-90.0` | Side view (orbit camera to one side of `Z`)                          |
| **Right**       | `0.0`     | `90.0`  | Opposite side                                                        |


**Important:** These are standard starting points; **Left/Right** may need a one-line swap of azimuth sign after a visual check in your scene. No change to `set_range` is required unless clipping appears (same as after mouse orbit).

## Implementation steps

1. **Imports:** Extend the existing `qtpy` import line to include `QtGui` (for `QAction` on Qt6) or use `from qtpy.QtGui import QAction` if preferred for clarity.
2. **Constants:** Define a small tuple/dict of presets next to the class or as module-level constants so **Perspective** stays in sync with the `TurntableCamera(...)` line (avoid duplicating `30`/`135` in two places—either reference the same constants in both `buildUI` and the menu handler, or set the initial camera from those constants).
3. **Helper method** on `Volumentric2DTimeSeriesPlotter`, e.g. `_apply_turntable_camera_preset(self, elevation: float, azimuth: float)`:
  - `cam = self.view.camera`; guard with `isinstance(cam, scene.TurntableCamera)`.
  - Set `cam.elevation`, `cam.azimuth`, and `cam.roll = 0.0`.
  - `self.canvas.update()` if `self.canvas` is not `None`.
4. **Menu wiring** in `buildUI`, immediately after `self.main_window = root_dockAreaWindow` (or just before `root_dockAreaWindow.show()`):
  - `mb = root_dockAreaWindow.menuBar()`
  - `view_menu = mb.addMenu("View")`  (or `"Camera"`)
  - For each preset, create `QAction(label, root_dockAreaWindow)`, connect `triggered` to a lambda/functor that calls `_apply_turntable_camera_preset` with the right `(elevation, azimuth)` (use default-arg closure to avoid Python late-binding bugs).
5. **No changes** to `[DockAreaWrapper.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/DockAreaWrapper.py)` unless you later want this menu on all dock windows (out of scope).

## Files touched

- Single file: [pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py) — `Volumentric2DTimeSeriesPlotter` only (~imports, `buildUI`, new helper).

