---
name: VispySceneWrappingWidget cleanup
overview: Restore `VispySceneWrappingWidget` to a small QWidget that embeds `canvas.native` with an optional horizontal splitter and `VispySceneTreeWidget`, then wire `Volumentric2DTimeSeriesPlotter.buildUI` to use it inside the existing Viewer dock (replacing duplicate scene-tree dock setup).
todos:
  - id: slim-wrap-widget
    content: "Replace VispySceneWrappingWidget in vispy_widgets.py: remove self-import, volumetric constants, pasted buildUI; implement splitter/native + optional tree, rebuild(), resizeEvent with qtpy QTimer; optional minimal canvas when None"
    status: completed
  - id: wire-plotter
    content: Update Volumentric2DTimeSeriesPlotter.buildUI in predicitive_decoding_vispy.py to embed VispySceneWrappingWidget, assign scene_tree_widget, remove duplicate Scene Tree dock
    status: completed
  - id: smoke-import
    content: Run quick import / linter check on edited files
    status: completed
isProject: false
---

# Finish VispySceneWrappingWidget for Volumentric2DTimeSeriesPlotter

## Problem

`[vispy_widgets.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py)` currently contains a **broken copy** of `[Volumentric2DTimeSeriesPlotter.buildUI](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` inside `VispySceneWrappingWidget.buildUI`: it references `self.n_t_bins`, `self.pos3d`, `_build_camera_view_menu`, `self.context`, `self.vispy_canvas`, etc., which do not exist on the widget. It also **imports `VispySceneWrappingWidget` from itself** (lines 402–403), which is invalid, and uses `CustomDockDisplayConfig` without importing it.

The real volumetric UI already lives in `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` (~1866–1960). The wrapper should not own docks, sliders, or 3D visuals.

## Target design

```mermaid
flowchart LR
  subgraph viewerDock [Viewer dock]
    sliders [t-bin / epoch sliders]
    wrap [VispySceneWrappingWidget]
    subgraph wrapInner [splitter when show_scene_tree]
      native [canvas.native]
      tree [VispySceneTreeWidget]
    end
    sliders --> wrap
  end
```



- `**VispySceneWrappingWidget**`: `QWidget` with `QVBoxLayout(0 margins)`. If `show_scene_tree` is false, add `canvas.native` with stretch 1. If true, horizontal `QSplitter`: `canvas.native` + `VispySceneTreeWidget(root_node=canvas.scene, canvas=canvas, column_renderers=...)`, stretch factors and `splitter_sizes` as in the **already-commented** block in `__init__` (lines 446–468). Expose `self.canvas`, `self.scene_tree_widget` (or `None`), and `rebuild()` delegating to the tree.
- **Do not** create `SceneCanvas.add_view()`, `VispyCanvasContainingWindow`, camera presets, or volumetric visuals here—the plotter (or other callers) owns the canvas and view.
- **Optional canvas**: If `canvas is None`, create a minimal `SceneCanvas(keys='interactive', show=False, ...)` and one default view with `view.camera = 'turntable'` so the widget is usable standalone without importing `CustomTurntableCamera` (keeps `vispy_widgets` free of volumetric-specific camera code). If `canvas` is supplied **with an existing central layout**, do not add another view (matches `Volumentric2DTimeSeriesPlotter`).
- **Resize**: Replace broken `on_resize` / `resizeEvent` with a single `resizeEvent` that calls `super()` then `QTimer.singleShot(10, self.canvas.update)` using `**qtpy.QtCore.QTimer`** (consistent with the file). Remove references to `self.context` and `self.vispy_canvas`.
- **Cleanup in this file**: Remove the erroneous mid-file imports (lines 402–410), duplicate `_VOLUMETRIC_`* constants (413–422), bogus self-import, and any unused `CustomTurntableCamera` import from `vispy_widgets` after the widget is slimmed down.

## Integrate into `Volumentric2DTimeSeriesPlotter.buildUI`

In `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`:

1. After `self.canvas` / `self.view` / camera are set up (unchanged), build `VispySceneWrappingWidget(canvas=self.canvas, parent=viewer_central_widget, show_scene_tree=True, tree_on_right=True, tree_minimum_width=200, column_renderers=..., splitter_sizes=...)` (forward `column_renderers` if you add a plotter `__init__` kwarg later; default `None` is fine).
2. Set `self.scene_tree_widget = wrapping.scene_tree_widget` so existing `rebuild()` call sites (~2199, ~2473) keep working.
3. Replace `viewer_layout.addWidget(canvas.native, stretch=1)` with `viewer_layout.addWidget(wrapping, stretch=1)`.
4. Remove the **separate** `VispySceneTreeWidget(...)` construction and the **second** `add_display_dock("Scene Tree", ...)` block; scene tree lives inside the Viewer dock via the splitter. Keep dock styling only for the Viewer dock (or add a second dock *only* if you introduce a flag—default plan is one dock for the combined viewer to match the widget’s purpose).

No change to `VispyCanvasContainingWindow` subclass in `vispy_widgets.py` beyond removing the stray pollution above it (the class stays as-is for other uses).

## Verification

- Import `pyphoplacecellanalysis.Pho2D.vispy.vispy_widgets` (no circular import).
- Instantiate `Volumentric2DTimeSeriesPlotter.init_from_position_and_decoder(...)` (smoke: window shows, GL + tree visible, sliders work, `scene_tree_widget.rebuild()` still safe).

