---
name: DataFrameRepresentable update_columns
overview: Add an `update_columns` method to the DataFrameRepresentable mixin with length checks and optional join/fill for mismatched lengths, plus a protected hook so Epoch/Laps can persist the updated dataframe in place.
todos: []
isProject: false
---

# Add update_columns to DataFrameRepresentable

## Context

- [dataframe_representable.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\dataframe_representable.py): mixin defines `to_dataframe()` only; no way to update backing data.
- [epoch.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py): `Epoch` stores backing data in `self._df`, implements `to_dataframe()` (returns `self._df.copy()`), no public setter.
- [laps.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\laps.py): `Laps` subclasses `Epoch`, same `_df` storage.

The mixin cannot assume `_df` exists (Position and others may use different attributes), so we need a small protocol: a protected method to write the updated dataframe back.

## Design

**1. New method on DataFrameRepresentable: `update_columns`**

- **Parameters**
  - `columns`: `Union[pd.DataFrame, Dict[str, pd.Series]]` — new columns to add or update (same names overwrite).
  - `join_on`: `Optional[str]` — column name to join on when lengths differ; must exist in both the current dataframe and in `columns` (or in its index if `columns` is a DataFrame with that index).
  - `fill_value`: `Any` (default `np.nan`) — value used for missing rows after a join when lengths differ.
  - `inplace`: `bool` (default `True`) — if `True`, write back via hook and return `None`; if `False`, return the updated dataframe without mutating.
- **Behavior**
  - Get current dataframe: `current = self.to_dataframe()`.
  - Normalize `columns` to a DataFrame (if dict, `pd.DataFrame(columns)`).
  - **Equal length**: if `len(current) == len(columns)`, assign new columns by integer position / reindex to `current.index` so alignment is well-defined; no `join_on` needed.
  - **Unequal length**: if `len(current) != len(columns)`:
    - If `join_on` is `None`, raise a clear `ValueError` (e.g. "Lengths differ; set join_on and optionally fill_value.").
    - Else: perform a left merge of `current` with `columns` on `join_on`, then fill any new columns’ missing values with `fill_value`.
  - If `inplace`: call `self._set_backing_dataframe(updated_df)` and return `None`. If not `inplace`: return `updated_df`.

**2. Protected hook: `_set_backing_dataframe(self, df: pd.DataFrame) -> None`**

- Add to **DataFrameRepresentable** with a default implementation that raises `NotImplementedError` and a short docstring: implementors that support in-place updates must override and assign their backing storage (e.g. `self._df = df`).
- **Epoch** (in [epoch.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py)): implement `_set_backing_dataframe(self, df)` with `self._df = df`. Then **Laps** inherits it and gets in-place column updates without touching private API.

**3. Safety checks (inside `update_columns`)**

- If lengths differ and `join_on` is `None`: raise `ValueError`.
- If `join_on` is provided: verify the column exists in `current` and in `columns` (or in `columns.index` if applicable) before merging; raise `ValueError` if missing.
- Use a left merge so row order and count of the current dataframe are preserved; only new columns are added/updated and filled where missing.

## Files to change


| File                                                                                                                                                        | Change                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [neuropy/utils/mixins/dataframe_representable.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\dataframe_representable.py) | Add `_set_backing_dataframe` (default `NotImplementedError`). Add `update_columns(columns, join_on=None, fill_value=np.nan, inplace=True)` with equal-length branch and join-on + fill branch, and optional typing imports. |
| [neuropy/core/epoch.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\epoch.py)                                                     | Implement `_set_backing_dataframe(self, df)` by setting `self._df = df` (one short method).                                                                                                                                 |


No changes to [laps.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\laps.py) required; Laps inherits from Epoch and will get both the mixin method and the Epoch hook.

## Usage example (after implementation)

```python
# Equal length: e.g. novelty columns computed per lap, same number of rows
laps_df = ensure_dataframe(a_sess.laps)
_, laps_df = compute_position_novelty(active_pos_df=pos_df, laps_df=laps_df)
new_cols = laps_df[['novelty_lehman_max', 'novelty_lehman_p90', 'novelty_knn_max', 'novelty_knn_p90']]
a_sess.laps.update_columns(new_cols, inplace=True)

# Or return updated df without mutating
updated_df = a_sess.laps.update_columns(new_cols, inplace=False)
a_sess.laps = Laps(updated_df, metadata=a_sess.laps.metadata)

# Unequal length: join on lap_id, fill missing with np.nan
a_sess.laps.update_columns(partial_novelty_df, join_on='lap_id', fill_value=np.nan, inplace=True)
```

## Edge cases to handle in implementation

- **columns as dict**: convert to DataFrame; if dict values are Series with index, merge path may use that index or a named column for `join_on`.
- **join_on with DataFrame**: if `columns` is a DataFrame, `join_on` can be a column name that exists in both `current` and `columns`; if `columns` is keyed by index (e.g. lap_id as index), document or support joining current’s `join_on` column to `columns.index` (e.g. `columns.reset_index()` then merge on `join_on`).
- **Overwrite existing columns**: when assigning new columns, same-name columns in `columns` overwrite those in current (pandas assign/merge behavior).

## Optional: Position

[Position](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py) also implements DataFrameRepresentable but uses a different backing store. It can override `_set_backing_dataframe` later if in-place column updates are desired; otherwise it can use `update_columns(..., inplace=False)` and then assign the result via its own API.