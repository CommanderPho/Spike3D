---
name: Left box tbin errors
overview: Thread per-bin held-out errors into the laps GT side-by-side figure and render them under the whole-lap median in the left metrics cell (1 decimal, small font).
todos:
  - id: wire-bins-df
    content: Pass train_test_pos_err_bins_df from batch completion → display → page plot
    status: completed
  - id: lookup-bins
    content: Add valid-bin error lookup matching metric_col (abs_err_cm_1D vs abs_err_cm)
    status: completed
  - id: draw-bins
    content: Annotate left cell with small 1-decimal bin list under median; widen left col slightly
    status: completed
isProject: false
---

# Show per-bin errors under left-box median

## Goal
For each lap row, keep the large whole-lap median (e.g. `136.5`) and add a smaller line underneath listing the **valid time-bin abs errors that entered that median**, each formatted to **1 decimal place**.

## Data already available
Per-bin long table is already computed as `train_test_pos_err_bins_df` in [`compute_kdiba_train_test_position_error_dfs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) / stashed by [`compute_and_export_kdiba_train_test_position_error_completion_function`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py).

Bin columns of interest: `lap_id`, `decoder_name`, `t`, `is_valid`, `abs_err_cm`, optional `abs_err_cm_1D`.

Epoch metric → bin column mapping:
- `abs_err_cm_1D_median` → `abs_err_cm_1D`
- `abs_err_cm_median` → `abs_err_cm`

Only bins with `is_valid == True` and finite bin error (same set that entered the median).

## Wiring
1. In [`figures_export_laps_correct_decoder_side_by_side_completion_function`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py): also pull `train_test_pos_err_bins_df` from the compute callback and pass it into `display(..., train_test_pos_err_bins_df=...)`.
2. In `_display_laps_correct_decoder_side_by_side_posteriors`: `kwargs.pop('train_test_pos_err_bins_df', None)` and forward through `_subfn_export_laps_four_decoder_gt_boxed_combined_figure` → `_subfn_plot_laps_four_decoder_posterior_page`.

## Lookup + draw (same file, nested helpers)
3. Add `_subfn_lookup_gt_pos_err_bin_values(bins_df, lap_id, truth_decoder_name, metric_col) -> List[float]`:
   - filter by decoder + lap
   - keep `is_valid` + finite values of the mapped bin column
   - sort by `t` / `time_bin_index`
4. Extend `_subfn_draw_pos_err_metrics_cell(..., bin_err_values: Optional[Sequence[float]] = None)`:
   - keep Lap label @ top and `test` @ bottom
   - move main median slightly up (`~0.62`)
   - if `bin_err_values` non-empty, draw under it (e.g. `y≈0.28`) as space-separated `f'{v:.1f}'` at fontsize ~5, same contrast color as the median, with wrapping (`wrap=True` / multiline string) so long test segments still fit
5. In the page loop, look up both the median and the bin list; pass both into the draw helper.

## Layout tweak
Slightly widen the left column (`width_ratios` from `[0.55, ...]` to about `[1.0, ...]`) so the bin list is readable without changing the rest of the page.

## Out of scope
No notebook edits. No change to how the median itself is computed.