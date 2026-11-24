---
name: Abs decoding error metrics
overview: Add per-epoch mean/median absolute decoding error (cm) alongside existing RMSE metrics in MeasuredDecodedPositionComparison, and propagate those columns into the KDIBA train/test position-error CSVs (per-epoch + session aggregates).
todos:
  - id: core-abs-metrics
    content: Add abs_err_cm_{mean,median} (+ 1D/2D variants) in build_single_measured_decoded_position_comparison
    status: completed
  - id: agg-export
    content: Extend compute_kdiba_train_test_position_error_dfs agg_kwargs + docstring for new columns
    status: completed
  - id: bapun-agg
    content: Mirror abs-error aggregations in BapunPositionDecodingPerformance
    status: completed
isProject: false
---

# Add mean/median absolute decoding error metrics

## Approach

Keep existing `sq_err` / `err_cm` (RMSE) unchanged. In the same per-epoch comparison, also compute literature-style **mean and median absolute error** over valid time bins, then aggregate those columns into the export CSVs.

## Core change

In [`build_single_measured_decoded_position_comparison`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (~L5311–5362):

For the primary 1D comparison (valid bins already used for MSE):

```python
abs_errs = np.abs(a_valid_decoded_positions - a_valid_interpolated_measured_x)
abs_err_cm_mean = float(np.mean(abs_errs))
abs_err_cm_median = float(np.median(abs_errs))
```

Add to `diff_row` (NaN when no valid bins):

- `abs_err_cm_mean`, `abs_err_cm_median`

For 2D (when computed), from per-bin Euclidean distances `np.sqrt(dx**2 + dy**2)`:

- `abs_err_cm_2D_mean`, `abs_err_cm_2D_median`

For optional native-1D `lin_pos` comparison (`*_1D`):

- `abs_err_cm_1D_mean`, `abs_err_cm_1D_median`

Update the method docstring to note these new columns.

`MeasuredDecodedPositionComparison` only stores the DataFrame — no schema change needed. All callers of `_do_custom_decode_epochs` / train–test decode pick this up automatically.

## Export aggregation

In [`compute_kdiba_train_test_position_error_dfs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (~L5855–5887):

- Per-epoch CSV (`train_test_pos_err`) already concatenates `decoded_measured_diff_df` — new columns appear automatically.
- Extend `agg_kwargs` for `train_test_pos_err_agg` to include mean/median of the new abs columns across epochs (mirror existing `err_cm_*` pattern), e.g. `abs_err_cm_mean_mean`, `abs_err_cm_mean_median`, `abs_err_cm_median_mean`, `abs_err_cm_median_median`, plus conditional `*_2D` / `*_1D` variants when those columns exist.
- Update the function docstring column list.

[`export_kdiba_train_test_position_error_csvs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) needs no logic change — it writes whatever columns the dfs contain.

## Bapun consistency (same comparison object)

In [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) `BapunPositionDecodingPerformance` aggregation (~L4571), add the same abs-error aggregations when those columns are present so Bapun exports stay aligned.

## Out of scope

- No confidence/posterior gating
- No renaming/removal of existing `err_cm` RMSE
- No notebook edits unless already driven by these helpers
