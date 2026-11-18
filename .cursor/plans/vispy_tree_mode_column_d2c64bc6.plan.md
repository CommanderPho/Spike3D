---
name: Vispy tree Mode column
overview: Add an editable "Mode" column to VispySceneTreeWidget that displays and allows editing the OpenGL render mode for vispy nodes that support it (e.g. Mesh mode, Line method/connect).
todos: []
isProject: false
---

# Add editable OpenGL render mode column to VispySceneTreeWidget

## Context

- [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py) defines `VispySceneTreeWidget` (lines 519–670) with columns: Type, Name, Visible, Order, Opacity, Transform. The only editable behavior today is the Visible checkbox (column 2) via `_on_item_changed`.
- Vispy visuals expose render/draw mode differently:
  - **Mesh**: has a settable `mode` attribute (`'triangles'`, `'triangle_strip'`, `'triangle_fan'`).
  - **Line**: has `method` (`'gl'` / `'agg'`) and `connect` (`'strip'` / `'segments'`); in the current environment `method` is settable, `connect` is not (frozen).
- Goal: add a **Mode** column that **displays** the render mode for nodes that support it, and is **editable** only where the backend allows (at least Mesh `mode`; Line `method` optional).

## Design

- Insert a new column **"Mode"** after **Opacity** and before **Transform** so that column indices become: 0 Type, 1 Name, 2 Visible, 3 Order, 4 Opacity, **5 Mode**, 6 Transform. Visible remains column 2.
- **Display** (read path):
  - **Mesh**: show `node.mode` (e.g. `triangles`, `triangle_strip`, `triangle_fan`).
  - **Line**: show `node.method + " / " + node.connect` (e.g. `gl / strip`).
  - Other nodes: show empty string.
- **Edit** (write path):
  - Only enable editing (e.g. `ItemIsEditable` for the Mode column) for nodes that have a settable mode. In practice: **Mesh** (set `node.mode`); optionally **Line** (set `node.method` only when user enters `gl` or `agg`).
  - On `itemChanged` for the Mode column: parse the new text; if it matches a valid value for that node type, set the attribute and call `canvas.update()`; on error, leave the node unchanged (and optionally revert the cell text on next rebuild).

## Implementation steps

1. **Add Mode to column list and UI**
  - In `__init`__: extend `_column_headers` to include `'Mode'` between `'Opacity'` and `'Transform'` (e.g. `['Type', 'Name', 'Visible', 'Order', 'Opacity', 'Mode', 'Transform']`).
  - In `_init_ui`: call `setColumnCount(7)` and add a resize mode for the new column (index 5), e.g. `ResizeToContents` like the other narrow columns.
2. **Default renderer for Mode**
  - In `_get_default_column_renderers`, add a `_render_mode(node)` that:
    - Returns `str(getattr(node, 'mode', ''))` for nodes that have `mode` (e.g. Mesh).
    - For Line (e.g. `isinstance(node, vz.Line)` or hasattr `method` and `connect`): return `f"{getattr(node, 'method', '')} / {getattr(node, 'connect', '')}"`.
    - Otherwise return `''`.
  - Register it under key `'Mode'` in the default renderers dict.
3. **Populate Mode column and editability**
  - In `_populate`: build the row list to include the mode string at index 5 (so the list has 7 elements: type, name, '', order, opacity, mode_text, transform_text).
  - Add a small helper (e.g. `_node_has_editable_mode(node)`) that returns True for Mesh (hasattr `mode`) and optionally for Line (hasattr `method`) so we only set editability where we can apply changes.
  - After creating the item, if `_node_has_editable_mode(node)` then set the item flags so that the **Mode column only** is editable. In Qt, tree items are per-item; to make only one column editable you typically set `ItemIsEditable` on the item and then in `_on_item_changed` ignore edits to other columns, or use a delegate. Simplest approach: set `item.setFlags(item.flags() | ItemIsEditable)` only when the node has editable mode; in `_on_item_changed` only apply when the changed column is 5 (Mode).
4. **Handle Mode edits in _on_item_changed**
  - In `_on_item_changed`: when `column == 5` (Mode), get the node from the item and the new text from `item.text(5)`.
  - If node has `mode` (Mesh): if new text is one of `('triangles', 'triangle_strip', 'triangle_fan')`, set `node.mode = new_text`.
  - If Line and node has `method`: if new text is `'gl'` or `'agg'`, set `node.method = new_text` (connect remains read-only).
  - After a successful set, call `self._canvas.update()` if `self._canvas` is not None. On exception, do nothing (next Refresh will restore displayed value).
5. **Qt compatibility**
  - Use the same pattern as existing code for Qt enums: e.g. `getattr(QtCore.Qt, 'ItemIsEditable', QtCore.Qt.ItemFlag.ItemIsEditable)` when setting editable flag.

## Files to change

- [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py): all changes in `VispySceneTreeWidget` and one new default renderer (and optional helper) in the same file. No new modules.

## Notes

- Keep the existing Visible checkbox behavior (column 2) unchanged.
- Optional: for Line, if we do not allow editing (because only `method` is settable and that might be confusing), we can still show "method / connect" in the Mode column and leave the cell non-editable for Line; the plan above allows either.
- Valid Mesh modes are the three vispy uses: `triangles`, `triangle_strip`, `triangle_fan`. Use case-insensitive matching if desired for robustness.

