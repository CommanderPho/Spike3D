---
name: Predictive dock window
overview: Refactor `PredictiveDecodingVispyWidget.buildUI` to embed the existing `SceneCanvas` in a `VispySceneWrappingWidget` inside a dock-area `VispyCanvasContainingWindow` (same structure as `Volumentric2DTimeSeriesPlotter`), fix early import ordering, and refresh the scene tree when visuals change.
todos:
  - id: fix-imports
    content: Move vispy_widgets imports before PredictiveDecodingVispyWidget; trim duplicate at ~1773
    status: completed
  - id: refactor-buildui
    content: Replace QMainWindow+canvas.native with DockAreaWrapper + VispySceneWrappingWidget + add_display_dock
    status: completed
  - id: scene-tree-rebuild
    content: Wire scene_tree_widget; add rebuild after buildUI and update_epoch_display (and multi-epoch path)
    status: completed
  - id: smoke-test
    content: Run single- and multi-epoch paths briefly; confirm window/slider/export still work
    status: completed
isProject: false
---

# PredictiveDecodingVispyWidget: dock + scene tree (match volumetric)

## Current state

- `[predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)` already declares `main_window: VispyCanvasContainingWindow` and `scene_tree_widget: VispySceneTreeWidget` on `PredictiveDecodingVispyWidget` (lines 144–145), but `**buildUI` still uses a plain `QtWidgets.QMainWindow**`, adds `canvas.native` directly, and never assigns `scene_tree_widget`.
- `VispyCanvasContainingWindow` / `VispySceneWrappingWidget` are imported **only after** this class (line 1773), while the class body references `VispyCanvasContainingWindow` in annotations—this is fragile (depends on PEP 563 / runtime) and should be corrected.

## Target pattern (reference)

`Volumentric2DTimeSeriesPlotter.buildUI` in the same file (approx. 1929–2027):

1. Create `SceneCanvas` as today.
2. `DockAreaWrapper.build_default_dockAreaWindow(title=..., defer_show=True)` → `root_dockAreaWindow` (note: factory returns `PhoDockAreaContainingWindow`; `VispyCanvasContainingWindow` is a thin subclass—optional improvement below).
3. Build `viewer_central_widget` + `QVBoxLayout`; add `VispySceneWrappingWidget(canvas=canvas, parent=viewer_central_widget, show_scene_tree=True, tree_on_right=True, tree_minimum_width=200, column_renderers=None, splitter_sizes=(700, 300))`; `**self.scene_tree_widget = scene_wrap.scene_tree_widget`** (assert non-None when tree is enabled).
4. Add epoch slider and table widgets to the **same** vertical layout (below the scene wrap), preserving existing behavior.
5. `CustomDockDisplayConfig(...)` matching volumetric (same `showCloseButton=False`, etc., and `build_custom_get_colors_fn` usage) then `root_dockAreaWindow.add_display_dock("Viewer", dockSize=..., widget=viewer_central_widget, dockAddLocationOpts=['left'], display_config=...)`.
6. `root_dockAreaWindow.resize(...)`, finish vispy grid/camera setup, `**self.scene_tree_widget.rebuild()`**, `root_dockAreaWindow.show()`.
7. `self.main_window = root_dockAreaWindow`.

No change to vispy scene logic (`canvas.central_widget.add_grid()`, views, `_render_central_view`, etc.) except **where** the native widget lives in Qt.

## Implementation steps

1. **Imports**
  - Add `from pyphoplacecellanalysis.Pho2D.vispy.vispy_widgets import VispySceneWrappingWidget, VispyCanvasContainingWindow` next to the existing vispy GUI imports (near `[DockAreaWrapper](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/DockAreaWrapper.py)` / `CustomDockDisplayConfig`, ~lines 84–87).  
  - Remove the redundant `VispySceneWrappingWidget, VispyCanvasContainingWindow` import from the pre-volumetric block (line 1773), keeping only `CustomTurntableCamera` from `vispy_cameras` there.
2. `**buildUI` refactor**
  - Replace `main_window = QtWidgets.QMainWindow()` / `setCentralWidget` / `addWidget(canvas.native)` with the dock + `VispySceneWrappingWidget` flow above.  
  - Keep canvas creation and all grid/view setup **after** the composite widget is in the layout (order can mirror volumetric: create canvas → build Qt shell → assign `self.canvas` → add wrap to layout → then `grid = canvas.central_widget.add_grid()` as today).
3. **Scene tree refresh**
  - Add a small `_refresh_scene_tree(self)` (or inline) that calls `self.scene_tree_widget.rebuild()` when not `None`.  
  - Invoke at end of `buildUI` (after first render path), and at end of `update_epoch_display` (and after the multi-epoch overview loop finishes), so the tree stays in sync when visuals are cleared/rebuilt—same motivation as `[Volumentric2DTimeSeriesPlotter._refresh_scene_tree](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)`.
4. **Optional strict typing**
  - If you want `self.main_window` to be a true `VispyCanvasContainingWindow` instance (not only the base dock window), instantiate `VispyCanvasContainingWindow(title=...)` with the same `defer_show` / `setWindowTitle` / `.app` pattern as `[DockAreaWrapper.build_default_dockAreaWindow](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/DockAreaWrapper.py)` instead of the factory. Otherwise, keep the factory exactly like volumetric for behavior parity (field type would be nominally wider than runtime class).
5. **Verification**
  - Smoke-test single-epoch mode: slider, tables (`enable_table_widgets`), keyboard epoch changes, `export_vispy_viewer_epochs` still sees `self.main_window` / `self.epoch_slider`.  
  - Smoke-test `enable_multi_epoch_overview_display_mode`: no epoch slider; ensure `buildUI` does not reference undefined locals (unchanged branch structure).

## Files touched

- Only `[predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)`: imports, `buildUI`, and minimal `_refresh_scene_tree` + call sites.

