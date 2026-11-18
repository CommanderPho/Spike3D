---
name: Filter update race fix
overview: Fix DataFrameFilter KeyErrors and torn state by publishing predicate counts via a local accumulator, serializing filter mutations with an RLock (callbacks outside the lock), and canceling pending debounce timers before sync updates.
todos:
  - id: add-rlock-field
    content: "Add `_update_lock: threading.RLock` field next to debounce_timer"
    status: completed
  - id: cancel-debounce-helper
    content: Add `_cancel_pending_debounce` and call from `update_filters`
    status: completed
  - id: local-counts-publish
    content: Refactor `update_filtered_dataframes` to local counts + single publish under RLock; callbacks outside lock
    status: completed
  - id: fix-size-table-assign
    content: Stop assigning to `filtered_size_info_df` property; use local + `table_widget.data`
    status: completed
isProject: false
---

# Fix DataFrameFilter concurrent update KeyError

## Problem
[`update_filtered_dataframes`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) repeatedly does `self.step_by_step_predicate_filtered_row_counts_dict[filtered_name].update(...)` while another debounce `Timer` call can replace that dict with `{}` at line 4013, causing `KeyError` mid-loop. Overlapping runs can also tear `_filtered_df_dict` / `is_filter_included` columns.

## Approach
Single-file change in [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py):

```mermaid
sequenceDiagram
    participant Widget
    participant Debounce as DebounceTimer
    participant Update as update_filtered_dataframes
    participant Lock as RLock
    participant UI as Table_Callbacks

    Widget->>Debounce: schedule _debounced_update
    Debounce->>Update: call
    Update->>Lock: acquire
    Note over Update: local counts dicts; mutate dfs; publish once
    Update->>Lock: release
    Update->>UI: callbacks + filename (outside lock)
```

### 1. Add re-entrant lock field
Near `debounce_timer` (~2784):

```python
_update_lock: threading.RLock = non_serialized_field(default=Factory(threading.RLock))
```

`threading` is already imported at module level (~2617). Use `RLock` so same-thread re-entry from a sync callback cannot deadlock.

### 2. Cancel pending debounce on sync entry
- Add a small helper `_cancel_pending_debounce(self)` that cancels `debounce_timer` if set and clears the reference.
- Call it at the start of [`update_filters`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) so notebook `df_filter.update_filters()` does not race a pending Timer.
- Keep existing cancel-before-reschedule behavior in `_on_widget_change`.

### 3. Rewrite count bookkeeping in `update_filtered_dataframes`
Inside the filter loop:

- Use locals only: `new_step_by_step: Dict[str, Dict[str, int]] = {}` and per-df `counts = {'META_n_unfiltered': len(df)}`.
- Write predicate / `CUM_*` / `META_post_filters` into `counts[...]` (direct assignment, not `self....update`).
- After each df is fully filtered: `new_step_by_step[filtered_name] = counts` and keep existing `_filtered_df_dict` / attrs logic.
- After the loop: **one** publish — `self.step_by_step_predicate_filtered_row_counts_dict = new_step_by_step`.
- Remove early `self.step_by_step_predicate_filtered_row_counts_dict = {}` and the dead `if filtered_name not in ...` init.

### 4. Lock scope (best practice for Jupyter)
Wrap **only** shared-state mutation + publish under `with self._update_lock:`:

- `output_widget` clear / filter loop / assign `step_by_step...` / build size table / set `table_widget.data` / fail warning print

Run **outside** the lock (after release):

- `on_filtered_dataframes_changed_callback_fns` loop
- `on_widget_update_filename()`

This avoids holding the lock across Plotly/widget callbacks while still serializing dataframe + counts updates.

### 5. Fix size-table assignment in the same method
Line 4110 currently assigns to property `filtered_size_info_df` (getter-only). Change to a local:

```python
merged_predicate_operations_df = self._build_merged_predicates_row_changed_df()
self.table_widget.data = merged_predicate_operations_df
```

(Do not shadow the property.)

## Out of scope
- No notebook edits
- No debounce redesign beyond cancel-on-sync
- Preserve existing predicate semantics and comments where still accurate

## Verify after change
In the notebook: recreate `DataFrameFilter` once (new `_update_lock` field), then rapidly change filters / call `update_filters()` and confirm no KeyError and table/plot still refresh.