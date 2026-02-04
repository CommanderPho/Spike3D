---
name: Dock title row/column count
overview: Add row and column counts to each table dock's title bar in StackedDynamicTablesWidget by setting the dock title when a table is added and whenever its data is updated, using the existing Dock.setTitle() API.
todos: []
isProject: false
---

# Show table row/column count in dock title

## Context

- [StackedDynamicTablesWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\Testing\StackedDynamicTablesWidget.py): `TableManager` creates docks via `add_display_dock(name, ...)` so the dock title is currently just the table name (e.g. `"Table_0"`).
- The pyqtgraph `Dock` class exposes `setTitle(text)` to set the displayed title; `name()` returns the internal identifier (unchanged). `find_display_dock(identifier)` matches on both `title()` and `name()`, so updating the displayed title does not break lookups by base name.

## Approach

Use a compact title format: `**{name} ({rows}×{cols})**` (e.g. `Table_0 (5×3)`). Set this in two places so it stays correct on add and on every update.

## Implementation

**1. Set title when adding a table** — in `add_table_dock` (after the dock is created and stored):

- After `self.dock_items[name] = dDisplayItem` (around line 272), call:
  - `dDisplayItem.setTitle(f"{name} ({df.shape[0]}×{df.shape[1]})")`
- This shows dimensions for newly created docks.

**2. Set title when updating a table** — in `_update_table` (after the model is refreshed):

- After `model = self._fill_table(table, df)` and the resize calls (before `return model`), call:
  - `dock_item.setTitle(f"{dock_item.name()} ({model.rowCount()}×{model.columnCount()})")`
- This keeps the title in sync when `update_tables` refreshes existing tables with new data.

No new helper is required unless you prefer one; the format is a single f-string in each place. No changes to `NestedDockAreaWidget`, `DynamicDockDisplayAreaContent`, or `Dock` are needed.

## Summary of edits


| Location         | Change                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| `add_table_dock` | After storing `dDisplayItem`, call `dDisplayItem.setTitle(f"{name} ({df.shape[0]}×{df.shape[1]})")`.                |
| `_update_table`  | Before `return model`, call `dock_item.setTitle(f"{dock_item.name()} ({model.rowCount()}×{model.columnCount()})")`. |


Result: each table dock’s title will show the table name plus current dimensions (e.g. `Table_0 (5×3)`), and the count will update whenever the table data is updated via `update_tables`.