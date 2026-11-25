---
name: GT-only brackets and I-bars
overview: Restrict train/test top brackets and measured-vs-decoded I-bars to the ground-truth (green-boxed) decoder subplot only, and source the I-bars directly from the valid rows of train_test_pos_err_bins_df so each bar matches a left-box tbin error 1-to-1.
todos:
  - id: shared-valid-bins-lookup
    content: Extract _subfn_lookup_gt_pos_err_valid_bins_df so left-box numbers and I-bars share the exact same valid-bin rows
    status: completed
  - id: gt-only-brackets-and-ibars
    content: Gate train/test brackets and valid-bin I-bars to the GT (green-boxed) decoder axis only and source I-bars from train_test_pos_err_bins_df
    status: completed
isProject: false
---

# GT-only train/test brackets and valid-bin I-bars

## Goal
In [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (`_display_laps_correct_decoder_side_by_side_posteriors`):
- Draw yellow/orange train/test top brackets **only** on the ground-truth decoder axis (`a_decoder_name == truth_name`, the green-boxed panel).
- Draw white/grey measured-vs-decoded I-bars **only** on the ground-truth decoder axis, sourced directly from the valid (`is_valid == True`) rows of `train_test_pos_err_bins_df` so the bars match the left-box tbin error list 1-to-1 (same decoder, same time bins, same valid-bin mask).

## Changes

### 1. Share one valid-bins lookup between left box and I-bars
In [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (around lines 11208-11233):
- Extract `_subfn_lookup_gt_pos_err_valid_bins_df(bins_df, lap_id, truth_decoder_name, metric_col) -> Optional[pd.DataFrame]` that:
  - filters `bins_df` by `decoder_name == truth_decoder_name` and `lap_id`
  - filters to `is_valid == True`
  - filters to finite values of the active error column (`abs_err_cm_1D` when `metric_col == 'abs_err_cm_1D_median'`, else `abs_err_cm`)
  - sorts by `['t', 'time_bin_index']` (stable mergesort)
- Have `_subfn_lookup_gt_pos_err_bin_values(...)` read the error column directly from `_subfn_lookup_gt_pos_err_valid_bins_df(...)` so both consumers see the exact same rows in the exact same order.

### 2. Gate brackets and I-bars to `is_gt_decoder = (a_decoder_name == truth_name)`
In `_subfn_plot_laps_four_decoder_posterior_page` (around lines 11294-11385):
- During the per-row lookup loop, retain `page_valid_bins_dfs: List[Optional[pd.DataFrame]]` alongside `page_err_values` and `page_bin_err_values`.
- Inside `for col_i, a_decoder_name in enumerate(decoder_names):`:
  - Only call `_subfn_shade_train_test_intervals(...)` when `(a_decoder_name == truth_name) and enable_train_test_spans and ((len(train_intervals) + len(test_intervals)) > 0)`.
  - Replace the per-column interpolation/I-bar block (lines 11369-11381) so that when `(a_decoder_name == truth_name)` and `row_valid_bins_df` is non-empty:
    - read `t_bins = row_valid_bins_df['t'].to_numpy(dtype=float)`
    - read `measured_col = 'measured_lin_pos' if (metric_col == 'abs_err_cm_1D_median' and 'measured_lin_pos' in row_valid_bins_df.columns) else 'measured'`
    - read `y_meas = row_valid_bins_df[measured_col].to_numpy(dtype=float)` and `y_dec = row_valid_bins_df['decoded'].to_numpy(dtype=float)`
    - call `_subfn_draw_test_interval_measured_decoded_ibars(ax, time_bin_centers=t_bins, measured_y=y_meas, decoded_y=y_dec, test_intervals=None)` (updating `_subfn_draw_test_interval_measured_decoded_ibars` so `test_intervals=None` draws all passed finite bins without re-filtering by interval boundaries, since `row_valid_bins_df` already contains exclusively the held-out test bins).
  - Non-GT decoder columns get neither train/test brackets nor test I-bars.
- Update the docstrings on `_display_laps_correct_decoder_side_by_side_posteriors`, `_subfn_plot_laps_four_decoder_posterior_page`, and `_subfn_export_laps_four_decoder_gt_boxed_combined_figure` to note that train/test brackets and valid-bin I-bars are rendered on the GT decoder column only.

## Verification
- Re-render the figure from the notebook and confirm:
  - Yellow/orange brackets appear only above the green-boxed GT panel on each row.
  - White/grey I-bars appear only on the green-boxed GT panel, and the number of I-bars on each lap equals the count of comma-separated tbin error values printed in the left box (e.g. 4 bars on Lap 4, 5 on Lap 5, 3 on Lap 6).