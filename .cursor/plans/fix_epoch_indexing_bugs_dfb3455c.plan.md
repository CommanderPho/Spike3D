---
name: Fix epoch indexing bugs
overview: Fix critical indexing bugs in `_subfn_filter_masked_container_epochs` where `epoch_idx_to_actual_times` lookup uses incorrect index mapping, and `iterrows()` returns DataFrame indices instead of positional indices.
todos:
  - id: fix-iterrows-positional
    content: Replace iterrows() DataFrame index with enumerate() positional index at line 3131
    status: completed
  - id: fix-epoch-times-lookup-1
    content: Fix epoch_idx_to_actual_times lookup to use filtered_idx at lines 3226-3228
    status: completed
  - id: fix-epoch-times-lookup-2
    content: Fix epoch_idx_to_actual_times lookup to use filtered_idx at lines 3240-3242
    status: completed
---

# Fix Epoch Indexing Bugs in `_subfn_filter_masked_container_epochs`

## Problem Summary

The function `_subfn_filter_masked_container_epochs` in [`PredictiveDecodingComputations.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) (lines 3082-3253) has two critical indexing bugs.

---

## Bug 1: Mismatched Index Domains for `epoch_idx_to_actual_times` Lookup

**Location:** Lines 3226-3228, 3240-3242

**Issue:** `epoch_idx_to_actual_times` is keyed by **filtered** epoch indices (0, 1, 2, ...) from `enumerate(time_bin_containers)`, but the code looks up using `original_idx` from the **original** DataFrame:

```python
original_idx = filtered_to_original_idx[filtered_idx] if ... else filtered_idx
if original_idx in epoch_idx_to_actual_times:
    epoch_start, epoch_stop = epoch_idx_to_actual_times[original_idx]
```

**Example of Bug:**

- Filtered epochs: indices 0, 1, 2, ... (50 epochs)
- `epoch_idx_to_actual_times = {0: times_0, 1: times_1, ..., 49: times_49}`
- `filtered_to_original_idx = [0, 3, 5, 8, ...]` (original indices)
- When `filtered_idx=0`: `original_idx=0`, lookup succeeds but returns correct times (by coincidence)
- When `filtered_idx=1`: `original_idx=3`, lookup returns times for filtered epoch 3, not filtered epoch 1

**Fix:** Use `filtered_idx` directly instead of `original_idx` for `epoch_idx_to_actual_times` lookup:

```python
if filtered_idx in epoch_idx_to_actual_times:
    epoch_start, epoch_stop = epoch_idx_to_actual_times[filtered_idx]
```

---

## Bug 2: `iterrows()` Returns DataFrame Index Labels, Not Positional Indices

**Location:** Lines 3130-3135

**Issue:** Using `orig_idx` from `original_active_epochs_df.iterrows()` as a list index. If the DataFrame has a non-sequential or non-integer index, this will produce incorrect results or errors:

```python
matching_original_idx = next((orig_idx for orig_idx, orig_row in original_active_epochs_df.iterrows()
    if ...), None)
# Then used as: pred_dec.matching_pos_epochs_dfs_list[i]
```

**Fix:** Use `enumerate()` to get positional indices:

```python
matching_original_idx = next((pos_idx for pos_idx, (_, orig_row) in enumerate(original_active_epochs_df.iterrows())
    if ...), None)
```

---

## Architectural Consideration: Redundant List Filtering

The filtering of `pred_dec.*_list` fields (lines 3138-3207) may be unnecessary because:

1. `compute_future_and_past_analysis` (called at line 3388) uses `override_included_analysis_epochs=a_masked_result.filter_epochs`
2. The lists are computed based on these **already-filtered** epochs
3. They should already be indexed correctly (0, 1, 2, ...) for filtered epochs

However, to maintain current behavior and minimize risk, the plan keeps this filtering but fixes the index usage.

---

## Implementation Changes

### Change 1: Fix `iterrows()` to use positional indices (Line 3131)

Replace:

```python
matching_original_idx = next((orig_idx for orig_idx, orig_row in original_active_epochs_df.iterrows()
    if (abs(row['start'] - orig_row['start']) < time_tolerance and abs(row['stop'] - orig_row['stop']) < time_tolerance) or
    (row['start'] <= orig_row['stop'] and row['stop'] >= orig_row['start'])), None)
```

With:

```python
matching_original_idx = next((pos_idx for pos_idx, (_, orig_row) in enumerate(original_active_epochs_df.iterrows())
    if (abs(row['start'] - orig_row['start']) < time_tolerance and abs(row['stop'] - orig_row['stop']) < time_tolerance) or
    (row['start'] <= orig_row['stop'] and row['stop'] >= orig_row['start'])), None)
```

### Change 2: Fix `epoch_idx_to_actual_times` lookup for `matching_pos_epochs_dfs_list` (Line 3226-3228)

Replace:

```python
original_idx = filtered_to_original_idx[filtered_idx] if (filtered_to_original_idx is not None and filtered_idx < len(filtered_to_original_idx)) else filtered_idx
if original_idx in epoch_idx_to_actual_times:
    epoch_start, epoch_stop = epoch_idx_to_actual_times[original_idx]
```

With:

```python
# Use filtered_idx directly since epoch_idx_to_actual_times is keyed by filtered indices
if filtered_idx in epoch_idx_to_actual_times:
    epoch_start, epoch_stop = epoch_idx_to_actual_times[filtered_idx]
```

### Change 3: Fix `epoch_idx_to_actual_times` lookup for `matching_pos_dfs_list` (Line 3240-3242)

Replace:

```python
original_idx = filtered_to_original_idx[filtered_idx] if (filtered_to_original_idx is not None and filtered_idx < len(filtered_to_original_idx)) else filtered_idx
if original_idx in epoch_idx_to_actual_times:
    epoch_start, epoch_stop = epoch_idx_to_actual_times[original_idx]
```

With:

```python
# Use filtered_idx directly since epoch_idx_to_actual_times is keyed by filtered indices
if filtered_idx in epoch_idx_to_actual_times:
    epoch_start, epoch_stop = epoch_idx_to_actual_times[filtered_idx]
```

---

## Testing Considerations

After applying fixes:

1. Verify that `is_future_present_past` column is correctly computed for all epochs
2. Check that list lengths remain consistent across all epoch-indexed structures
3. Test with DataFrames that have non-sequential indices to ensure positional indexing works