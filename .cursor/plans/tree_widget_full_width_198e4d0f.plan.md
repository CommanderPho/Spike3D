---
name: Tree widget full width
overview: Make VispySceneTreeWidget expand to fill the full width of its parent dock/container, both internally (column stretch) and externally (size policy, removal of max-width cap at the usage site).
todos:
  - id: stretch-last
    content: Change `setStretchLastSection(False)` to `setStretchLastSection(True)` on line 602 of vispy_helpers.py
    status: completed
  - id: size-policy
    content: Add expanding QSizePolicy to the VispySceneTreeWidget itself in `_init_ui`
    status: completed
  - id: remove-max-width
    content: Remove `setMaximumWidth(320)` from predicitive_decoding_vispy.py line 1827
    status: completed
isProject: false
---

# Tree Widget Full Width Fix

## File 1: [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)

Two changes in `_init_ui` (around lines 596-611):

- **Line 602:** Change `setStretchLastSection(False)` to `setStretchLastSection(True)` so the last column (Transform) stretches to fill remaining horizontal space.
- **Add expanding size policy** to the widget itself so it requests all available space from its parent layout/dock:

```python
expanding = getattr(QtWidgets.QSizePolicy, 'Expanding', QtWidgets.QSizePolicy.Policy.Expanding)
self.setSizePolicy(QtWidgets.QSizePolicy(expanding, expanding))
```

## File 2: [predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)

In `Volumentric2DTimeSeriesPlotter.buildUI()` (lines 1826-1828):

- **Remove line 1827** (`self.scene_tree_widget.setMaximumWidth(320)`) -- this hard cap prevents the widget from filling wider docks when the user resizes the window. The `setMinimumWidth(200)` on line 1828 is fine to keep; it prevents the dock from collapsing too small.

No other usage sites need changes: `example_scene_tree_widget()` and `create_scene_tree_widget()` don't impose width constraints.