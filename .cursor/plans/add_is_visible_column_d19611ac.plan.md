---
name: Add is_visible column
overview: Add an optional boolean column `is_visible` to the IntervalsDatasource dataframe, defaulting to all True when not provided, by ensuring the column in `__init__` only.
todos: []
isProject: false
---

# Add `is_visible` bool column to IntervalsDatasource

## Summary

Add a boolean column `is_visible` to the `IntervalsDatasource` dataframe. When the column is not present on the incoming dataframe, it will be added with all values `True`; when provided by the caller, it will be left unchanged.

## Approach

**Single change in [IntervalDatasource.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\Datasources\IntervalDatasource.py):**

In `__init__` (lines 131–142), after the dataframe is validated and assigned to `self._df`, ensure the `is_visible` column exists:

- **If `'is_visible' not in self._df.columns**`: add `self._df['is_visible'] = True` (pandas will broadcast to all rows).
- **If `'is_visible'` is already present**: do nothing, preserving caller-provided values.

This keeps all construction paths correct:

- **Direct construction** `IntervalsDatasource(df, ...)`: column added when missing.
- **Factory** `init_from_epoch_object` / `init_from_times_values`: they call `cls(active_df, ...)`, so the same logic applies; no changes needed in those classmethods.
- **Caller-provided** `is_visible`: any dataframe passed in with `is_visible` already set is left as-is.

## Optional documentation

You may add a class-level comment or include `'is_visible'` in an optional-columns list (e.g. near `_required_interval_visualization_columns` around line 49) for discoverability; the plan does not require it.

## Out of scope

- No changes to `get_serialized_data` / `get_deserialized_data` (they handle pen/brush only).
- No changes to `get_updated_data_window` or rendering code; future callers can filter with `df[df['is_visible']]` when they need visibility-aware behavior.

