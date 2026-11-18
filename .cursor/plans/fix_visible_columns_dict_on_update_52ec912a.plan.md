---
name: Fix visible_columns_dict on update
overview: Restore visible_columns_dict behavior on table updates by applying the current visible_columns_dict for each table inside _update_table before replacing the model, so that initCustomHeaders uses the correct column list when the new model is set.
todos: []
isProject: false
---

# Fix visible_columns_dict maintenance on table update

## Root cause

Column visibility on **update** is broken because `_update_table` never uses `visible_columns_dict`:

- **add_table_dock** (lines 256–259): Reads `self.visible_columns_dict.get(name, ...)`, computes `included_visible_columns`, and passes it to `_create_table(..., visible_columns=included_visible_columns)`. The table stores that and `CustomHeaderTableView.initCustomHeaders()` applies it when the model is set.
- **update_tables** (231–236): Merges any passed `visible_columns_dict` into `self.visible_columns_dict`, then for existing tables calls `_update_table(self.dock_items[name], df)`.
- **_update_table** (331–348): Only calls `_fill_table(table, df)` (which does `setModel(new_model)`). It never reads `self.visible_columns_dict` or sets `table.visible_columns`. So when `setModel` triggers `initCustomHeaders()`, the table still has its **original** `visible_columns` from creation. Any later changes in `visible_columns_dict` (or updates that add/remove columns) are ignored for existing tables.

The dock title change did not cause this; it only made the issue more noticeable. The fix is to make `_update_table` apply the current `visible_columns_dict` before replacing the model.

## Approach

In `_update_table`, before calling `_fill_table(table, df)`:

1. Resolve the table name: `name = dock_item.name()` (same key as in `dock_items`).
2. Get the current visible columns for that name: `visible_columns = self.visible_columns_dict.get(name, None)` (or use the same default as `add_table_dock` if you want consistency: `['start', 'delta_aligned_start_t', 'label', 'unique_active_aclus']`).
3. Compute included visible columns for the **current** `df`: `included_visible_columns = [col for col in df.columns if col in visible_columns]` when `visible_columns` is not None; otherwise leave as None so `initCustomHeaders` can default to all visible.
4. Set the table’s visibility list: `table.visible_columns = included_visible_columns if (included_visible_columns is not None and len(included_visible_columns) > 0) else None`.
5. Then call `_fill_table(table, df)` as now. `setModel` will run `initCustomHeaders()`, which will use the updated `table.visible_columns`.

This mirrors the logic in `add_table_dock` (same default and same `included_visible_columns` computation) so that add and update behave consistently.

## File and edits

**File:** [StackedDynamicTablesWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\Testing\StackedDynamicTablesWidget.py)

**In `_update_table**` (method starting ~line 331):

- After resolving `table` from `dock_item.widgets[0]`, and **before** `model = self._fill_table(table, df)`:
  - Add: resolve `name = dock_item.name()`.
  - Add: `visible_columns = self.visible_columns_dict.get(name, ['start', 'delta_aligned_start_t', 'label', 'unique_active_aclus'])` (same default as `add_table_dock`).
  - Add: `included_visible_columns = [col for col in df.columns if col in visible_columns]`.
  - Add: `table.visible_columns = included_visible_columns if included_visible_columns else None`.
- Leave the rest unchanged (`_fill_table`, resize, setTitle, return).

No changes to `add_table_dock`, `update_tables`, or `CustomHeaderTableView` are required.

## Result

- When an existing table is updated via `update_tables`, its visible columns will be driven by the current `self.visible_columns_dict` for that table name.
- New tables continue to use `visible_columns_dict` at creation as before.
- Dock title (row/column count) behavior is unchanged.

