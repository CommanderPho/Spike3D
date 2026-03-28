---
name: VispyCanvasContainingWindow minimal
overview: Fix `VispyCanvasContainingWindow` so it correctly forwards construction to `PhoDockAreaContainingWindow`, remove stale comments, and align the docstring with a vispy + dock-area window (without adding factories or vispy-specific logic yet).
todos:
  - id: init-forward-title
    content: Fix __init__ to pass title to super; optionally setWindowTitle(title)
    status: completed
  - id: docstring-cleanup
    content: Replace misleading Usage with vispy+dock-area description; remove stale comments
    status: completed
  - id: smoke-import
    content: Run minimal import/instantiation check
    status: completed
isProject: false
---

# Minimal `VispyCanvasContainingWindow` implementation

## Context

- `[VispyCanvasContainingWindow](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py)` subclasses `[PhoDockAreaContainingWindow](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockAreaWrapper.py)`, which already:
  - Calls `DynamicDockDisplayAreaContentMixin_on_init()`, then `PhoMainAppWindowBase.__init__`, then `setup()` (creates `DockArea`, `setCentralWidget`), then `buildUI()`.
  - Exposes `.area` and mixin methods such as `add_display_dock` for embedding widgets.
- Current stub calls `super(VispyCanvasContainingWindow, self).__init__(*args, **kwargs)` only, so the explicit `title` parameter is **never passed** to the parent (same omission pattern as `[PhoPaginatedMultiDecoderDecodedEpochsWindow.__init_](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py)_` line 1995–1996, but fixing it here is low-cost and keeps the API honest).
- The commented `# self.setup()` / `# self.buildUI()` and the `# def draw` block are wrong for this type: setup is already done by the parent; `draw()` referred to matplotlib paginated widgets and does not apply.

## Planned edits (single file)

**File:** `[vispy_widgets.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py)` (`VispyCanvasContainingWindow` only)

1. `**__init__`** — Forward `title` to the parent so the signature matches `[PhoDockAreaContainingWindow.__init_](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockAreaWrapper.py)_` (first positional `title`, then `*args`, `**kwargs`):
  - `super(VispyCanvasContainingWindow, self).__init__(title, *args, **kwargs)`
2. **Optional but reasonable for a “window” type:** `self.setWindowTitle(title)` so a bare `VispyCanvasContainingWindow()` shows the intended title without every call site duplicating `[DockAreaWrapper.build_default_dockAreaWindow](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\DockAreaWrapper.py)`’s pattern. (If you prefer to match `PhoDockAreaContainingWindow` exactly and always set the title at call sites, skip this line; the plan defaults to including it for usability.)
3. **Docstring** — Replace the copied pagination/decoder **Usage** block with a short, accurate note: this is a dock-area main window for embedding vispy content (typically `SceneCanvas.native` or a `VispySceneWrappingWidget`) via `add_display_dock`; point to `PhoDockAreaContainingWindow` / `DockAreaWrapper` for general dock patterns.
4. **Delete** the obsolete commented `setup`/`buildUI` and the commented `draw` method block to keep the class minimal and avoid implying double-setup or matplotlib redraw.

## Out of scope (explicitly not in “minimal”)

- No new `build_default_`* classmethod unless you later want parity with `DockAreaWrapper.build_default_dockAreaWindow` but typed as `VispyCanvasContainingWindow`.
- No automatic vispy canvas creation; callers continue to create `SceneCanvas` and dock `canvas.native` (consistent with existing code in `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`).

## Verification

- Smoke: `python -c` import and `VispyCanvasContainingWindow(title='test')` (no need to show a window if headless); assert `hasattr(win, 'area')` and window title matches if `setWindowTitle` is included.

