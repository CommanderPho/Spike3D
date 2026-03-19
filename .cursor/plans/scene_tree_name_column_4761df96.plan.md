---
name: Scene tree Name column
overview: Reorder `VispySceneTreeWidget` columns so **Name** is first and **Type** is second, keep all other column indices unchanged, and reinforce auto-width for the Name column (including a post-rebuild resize pass).
todos:
  - id: swap-headers-populate
    content: Update _column_headers and QTreeWidgetItem column order in _populate
    status: completed
  - id: rebuild-resize-name
    content: Call resizeColumnToContents(0) at end of rebuild(); confirm header modes still map to correct columns
    status: completed
isProject: false
---

# VispySceneTreeWidget: Name first + auto-size

## Context

`[vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)` defines `VispySceneTreeWidget` with seven columns. Today the order is `Type`, `Name`, `Visible`, … and `_populate` builds items as `[node_type, node_name, …]`. **QTreeWidget always draws the tree branch controls on column 0**, so after this change the expand/collapse affordance will sit next to **Name** (standard pattern for layer/scene trees).

Functional columns that use hard-coded indices **stay the same** after swapping only the first two columns:


| Index | Before                   | After     |
| ----- | ------------------------ | --------- |
| 0     | Type                     | **Name**  |
| 1     | Name                     | **Type**  |
| 2     | Visible (checkbox)       | Visible   |
| 5     | GL Blend (delegate)      | GL Blend  |
| 6     | Transform (stretch last) | Transform |


So `[_on_item_changed](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)` (`column in (2, 5)`), `setCheckState(2, …)`, and `setItemDelegateForColumn(5, …)` require **no updates**.

## Implementation (single file)

**File:** `[pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)`

1. `**_column_headers`** — set to `['Name', 'Type', 'Visible', 'Order', 'Opacity', 'GL Blend', 'Transform']`.
2. `**_populate`** — build the item as `[node_name, node_type, '', node_order, node_opacity, gl_blend_text, transform_text]` so cell order matches headers. Keep `item.setData(0, self._user_role, node)` (node ref remains on first column).
3. **Header sizing (`_init_ui`)** — column 0 remains `ResizeToContents` and becomes the Name column; column 1 remains `ResizeToContents` for Type. `setStretchLastSection(True)` unchanged (Transform still consumes extra width). No index changes needed for modes 2–6.
4. **Stronger Name auto-width after rebuild** — at the end of `rebuild()`, after `expandToDepth(3)`, call `resizeColumnToContents(0)` on `self.tree` so Qt recomputes width from **all visible rows** (not only the header label). This addresses “fit the names listed” more reliably than layout alone when the tree is first filled.

No other call sites reference column numbers for this widget; `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` only imports the class.

## Verification

- Manually open the scene tree UI: first column shows names with branches; second shows types; checkboxes and blend editor still work.
- Optional: run the existing example `example_scene_tree_widget()` if you want a quick smoke test.

