---
name: Hide Type Transform columns
overview: Hide the "Type" (index 1) and "Transform" (index 6) columns by default in `VispySceneTreeWidget` using Qt's built-in column hiding, with a short doc note so users can re-show them if needed.
todos:
  - id: hide-columns
    content: Call setColumnHidden(1) and setColumnHidden(6) in VispySceneTreeWidget._init_ui after tree setup; optional _COL_* constants
    status: completed
  - id: doc-note
    content: Document default-hidden Type/Transform and how to show via setColumnHidden(False)
    status: completed
isProject: false
---

# Hide Type and Transform columns by default

## Context

In `[vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)`, `VispySceneTreeWidget` uses seven columns in fixed order:


| Index | Column    |
| ----- | --------- |
| 0     | Name      |
| 1     | Type      |
| 2     | Visible   |
| 3     | Order     |
| 4     | Opacity   |
| 5     | GL Blend  |
| 6     | Transform |


No structural change to `_populate`, renderers, or `_on_item_changed` is required: hidden columns still exist logically; only the view hides them.

## Implementation

1. **After the tree is configured in `_init_ui`** (after header resize modes and delegate on column 5, before `layout.addWidget`), call:
  - `self.tree.setColumnHidden(1, True)`  — Type  
  - `self.tree.setColumnHidden(6, True)`  — Transform
   Optional but clearer: define module- or class-level integer constants for these indices (e.g. `_COL_TYPE = 1`, `_COL_TRANSFORM = 6`) and use them in `setColumnHidden` so future column edits are less error-prone.
2. **Docstring** — Add one sentence to the class docstring (near the existing Transform/renderer notes) stating that Type and Transform are hidden by default and can be shown again with `tree.setColumnHidden(<index>, False)` (or the constant names if added).

## Notes

- `rebuild()` uses `tree.clear()` only; it does **not** reset column hidden state, so defaults stay applied across refreshes.
- `resizeColumnToContents` for columns 1 and 6 in `rebuild()` remains valid; Qt typically no-ops or handles hidden sections safely. No change required unless you observe layout quirks (unlikely).

## Scope

- Single file: `[vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)` (`VispySceneTreeWidget` only).

