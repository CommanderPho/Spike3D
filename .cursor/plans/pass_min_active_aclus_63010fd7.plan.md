---
name: Pass min active aclus
overview: Wire `min_num_unique_active_neurons_per_time_bin` (default 1) through the train-test position-error and laps side-by-side figure export paths so `--only` recipes / completion kwargs can set it.
todos:
  - id: thread-pos-err
    content: Thread min_num_unique_active_neurons_per_time_bin=1 through decode/eval → export CSV → completion_function
    status: completed
  - id: align-side-by-side-default
    content: Change laps side-by-side display/completion default from 3 to 1
    status: completed
  - id: batch-kwargs
    content: Add param to ProcessBatchOutputs shared kwargs (and recipe override kwargs for the two --only recipes)
    status: completed
isProject: false
---

# Pass `min_num_unique_active_neurons_per_time_bin` through batch exports

**Assumption:** You meant the two masking-related `--only` recipes — `export_train_test_pos_err` (alias at line 273) and `figures_export_laps_correct_decoder_side_by_side` (alias at line 271) — not `figures_export_2d_position_over_time` (line 274), which has no firing-aclu mask. Default for this param will be **`1`** as specified (overriding the current hard-coded `3` on these paths).

## 1. Thread param through train-test pos-err stack

Today masking lives only in [`_do_custom_decode_epochs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (`min_num_unique_active_neurons_per_time_bin=3`), but callers do not expose it.

Pass `min_num_unique_active_neurons_per_time_bin: Optional[int] = 1` (and keep existing `should_mask_by_required_spike_counts` / spike-count kwargs) through:

- `_do_train_test_split_decode_and_evaluate` → `_do_custom_decode_epochs` / `_do_custom_decode_epochs_dict` via `**kwargs` or explicit args
- `compute_kdiba_train_test_position_error_dfs`
- `export_kdiba_train_test_position_error_csvs`
- [`compute_and_export_kdiba_train_test_position_error_completion_function`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) (signature + forward into `export_kdiba_train_test_position_error_csvs`)

Change `_do_custom_decode_epochs` default for this param from `3` → `1` so all callers stay consistent unless overridden.

## 2. Align laps side-by-side figure defaults

[`figures_export_laps_correct_decoder_side_by_side_completion_function`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) and `_display_laps_correct_decoder_side_by_side_posteriors` already accept the param; change their defaults from `3` → `1`.

## 3. Wire kwargs in ProcessBatchOutputs

In [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy):

- Add to shared kwargs dict (~line 554):
  - `'compute_and_export_kdiba_train_test_position_error_completion_function': dict(..., min_num_unique_active_neurons_per_time_bin=1)`
  - `'figures_export_laps_correct_decoder_side_by_side_completion_function': dict(min_num_unique_active_neurons_per_time_bin=1)` (new entry if missing)
- Optionally add the same under each recipe’s `completion_fn_override_kwargs` for `export_train_test_pos_err` and `figures_export_laps_correct_decoder_side_by_side` so `--only` runs get an explicit, documented value.

No CLI flag unless you want one later; completion kwargs already flow through `custom_user_completion_function_override_kwargs_dict`.

## Out of scope

- `figures_export_2d_position_over_time` (line 274 alias) — no decoder / firing-aclu mask.