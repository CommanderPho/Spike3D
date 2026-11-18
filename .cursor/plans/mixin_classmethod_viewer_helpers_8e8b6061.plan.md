---
name: Mixin classmethod viewer helpers
overview: Move the three module-level viewer/dock helpers onto `VispySceneWindowMixin` as `@classmethod`s, remove the loose functions, fix `VispySceneWrappingWidget.resizeEvent` accidentally nested under the old helper, and switch call sites in `predicitive_decoding_vispy.py` to use the mixin (via `self`).
todos:
  - id: mixin-classmethods
    content: Add three @classmethod helpers to VispySceneWindowMixin; wire add_standard default to cls.default_...
    status: completed
  - id: fix-wrap-resize
    content: Move resizeEvent into VispySceneWrappingWidget; delete loose defs + orphaned nested resizeEvent
    status: completed
  - id: wire-predictive-file
    content: Update predicitive_decoding_vispy imports and 4 buildUI calls to self.*
    status: completed
  - id: verify-compile
    content: py_compile both modules; grep for stale function imports
    status: completed
isProject: false
---

# Refactor viewer helpers into `VispySceneWindowMixin`

## Context

- The three functions at [vispy_widgets.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py) lines 579–596 sit **between** `[VispySceneWrappingWidget.rebuild](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py)` and what was meant to be `resizeEvent`.
- Because `resizeEvent` (lines 600–603) is indented inside `add_standard_vispy_viewer_dock`, it is a **nested function**, not a `VispySceneWrappingWidget` method. That should be corrected while touching this region.

## Implementation

### 1. Add three `@classmethod` helpers on `VispySceneWindowMixin`

In [vispy_widgets.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py), after `_bind_scene_tree_from_wrap` (around line 428), add:

- `default_vispy_viewer_dock_display_config(cls) -> CustomDockDisplayConfig` — same body as today’s function.
- `make_viewer_central_widget_with_scene_wrap(cls, canvas, *, ...)` — same body; instantiate `VispySceneWrappingWidget(...)` as today (safe at **call time** because the class is defined later in the module).
- `add_standard_vispy_viewer_dock(cls, main_window, viewer_central_widget, *, ...)` — same body, but default config via `cls.default_vispy_viewer_dock_display_config()` instead of the module-level name.

Use `@classmethod` and keep signatures on one line where feasible ([user rule](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D.cursor\rules\User-requires-function-signatures-to-be-on-a-single-line-when-possible-rather-than-split-across-multiple-lines.mdc)); if `make_viewer_central_widget_with_scene_wrap` exceeds line length, split only if necessary.

Maintain **two blank lines** between methods on the mixin ([user preference](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D.cursor\rules\User-prefers-two-blank-lines-between-python-class-methods-for-readibility.mdc)).

### 2. Restore `VispySceneWrappingWidget.resizeEvent` and delete module-level defs

- After `rebuild` in `VispySceneWrappingWidget`, insert the `resizeEvent` method (same implementation as the stray block at 600–603).
- Delete the three standalone `def default_...` / `def make_...` / `def add_...` functions and the orphaned nested `resizeEvent` block (current lines ~579–603).

### 3. Update imports and call sites

Only external usage is [predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py):

- **Import line (~87):** Drop `make_viewer_central_widget_with_scene_wrap` and `add_standard_vispy_viewer_dock`; keep `VispySceneWindowMixin` (already imported).
- `**PredictiveDecodingVispyWidget.buildUI` (~286, ~320):** Replace `make_viewer_central_widget_with_scene_wrap(...)` with `self.make_viewer_central_widget_with_scene_wrap(...)` and `add_standard_vispy_viewer_dock(...)` with `self.add_standard_vispy_viewer_dock(...)`.
- `**Volumentric2DTimeSeriesPlotter.buildUI` (~1915, ~1956):** Same replacements.

Calling via `self` is correct for `@classmethod`s on the mixin.

### 4. Verify

- `python -m py_compile` on `vispy_widgets.py` and `predicitive_decoding_vispy.py`.
- Quick grep that the old free-function names no longer appear as imports except as class attributes.

## Files touched


| File                                                                                                                                                                          | Change                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [vispy_widgets.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_widgets.py)                           | Classmethods on mixin; remove loose functions; fix `resizeEvent` on `VispySceneWrappingWidget` |
| [predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py) | Import + 4 call sites                                                                          |


