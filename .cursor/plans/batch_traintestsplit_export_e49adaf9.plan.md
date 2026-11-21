---
name: Batch TrainTestSplit Export
overview: Ensure KDIBA phase3 computes TrainTestSplit (already listed), then decode held-out lap test segments, export position-error CSVs in collected_outputs parseable form next to perfmnc, and wire local loading.
todos:
  - id: ensure-eval-export
    content: Add KDIBA TrainTestSplit ensure + _do_train_test_split_decode_and_evaluate → train_test_pos_err / _agg CSV builder
    status: completed
  - id: wire-generalized-decode
    content: Call export from generalized_decode_epochs_dict_and_export_results_completion_function after default_export_all_CSVs
    status: completed
  - id: local-load
    content: Register train_test_pos_err and train_test_pos_err_agg in _new_process_csv_files extra_df sinks
    status: completed
isProject: false
---

# Batch TrainTestSplit position-error export

## Context

- **`directional_train_test_split` is already in KDIBA phase3** ([`pythonScriptTemplating.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py) ~325) → `computed_data['TrainTestSplit']`.
- **`perfmnc_session` CSVs** (context % correct) export from `generalized_decode_epochs_dict_and_export_results_completion_function` → `default_export_all_CSVs` → `export_csvs` (~2294 in [`context_dependent.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\context_dependent.py)).
- **No KDIBA CSV export** yet for held-out **position** error. Bapun has a separate path; do not reuse its non-parseable filenames for KDIBA.
- Evaluate path already exists: [`_do_train_test_split_decode_and_evaluate`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) (~5705) with `compute_separate_decoder_results=True` → `train_decoded_measured_diff_df_dict` (`t`, `sq_err`, `err_cm`, … per decoder).

```mermaid
flowchart LR
  Phase3["phase3: directional_train_test_split"] --> TTS["computed_data TrainTestSplit"]
  TTS --> Eval["_do_train_test_split_decode_and_evaluate"]
  Eval --> DiffDF["per-decoder decoded_measured_diff_df"]
  DiffDF --> CSV["collected_outputs train_test_pos_err CSVs"]
  CSV --> Load["_new_process_csv_files local load"]
```

## Approach (locked)

Extend **`generalized_decode_epochs_dict_and_export_results_completion_function`** in [`batch_user_completion_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) immediately after the existing `default_export_all_CSVs` block (~3110), so it runs in the same KDIBA final_run job as perfmnc. Reuse the dead `TrainTestSplit*` imports already present (~2974).

Default decode bin size: **`epochs_decoding_time_bin_size`** already used by that completion (same `_tbin-` string as other exports). Overridable via a new kwarg `train_test_laps_decoding_time_bin_size`.

## Implementation

### 1. Batch export helper (KDIBA)

In [`batch_user_completion_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) (or a small pure helper in [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) if cleaner):

1. Ensure `TrainTestSplit` exists (call `perform_specific_computation(['directional_train_test_split'])` if missing).
2. Call `_do_train_test_split_decode_and_evaluate(..., compute_separate_decoder_results=True, active_laps_decoding_time_bin_size=...)`.
3. Build export frames from `_out_separate_decoder_results`:
   - **`train_test_pos_err`**: concat each decoder’s `decoded_measured_diff_df` with columns `decoder_name`, `session_name`, `time_bin_size`, plus existing `t` / `sq_err` / `err_cm` (and `*_1D`/`*_2D` if present).
   - **`train_test_pos_err_agg`**: per-`decoder_name` mean/median `err_cm` (and counts) for quick session QC.
4. Write via the same session-identifier CSV naming as perfmnc (`build_complete_session_identifier_filename_string` / `_perform_export_dfs_dict_to_csvs` pattern) with:
   - `an_override_df_identifier='train_test_pos_err'`
   - `an_override_df_identifier='train_test_pos_err_agg'`
   - `_tbin-{size}` suffix so `parse_filename` yields `file_type` correctly.

Wrap in try/except so a TrainTestSplit failure does not abort the rest of generalized_decode CSV export (log + store error in `across_session_results_extended_dict`).

### 2. Wire into `generalized_decode_...` completion

After `csv_save_paths_dict = a_new_fully_generic_result.default_export_all_CSVs(...)` (~3110):

- Run the helper.
- Merge returned paths into `csv_save_paths_dict` / callback outputs under keys like `train_test_pos_err_csv_paths`.

No change needed to `phase3_extended_computations_include_includelist` (already has `directional_train_test_split`). No new phase3 completion registration required.

### 3. Local import path

In [`AcrossSessionResults.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py) `_new_process_csv_files` (~3532):

- Add `'train_test_pos_err'` and `'train_test_pos_err_agg'` to `final_sessions_loaded_extra_df_dict` / `final_sessions_loaded_extra_dfs`.
- Handle like `perfmnc_session` / `fig5_occupancy`: `time_key=None` (or `'t'` for the detailed DF if preferred for delta alignment).

After a batch + `load_across_sessions_exported_files`, local notebook access:

```python
train_test_pos_err_df = final_sessions_loaded_extra_dfs_dict['train_test_pos_err']
train_test_pos_err_agg_df = final_sessions_loaded_extra_dfs_dict['train_test_pos_err_agg']
```

Add a short usage note / one-liner next to `PerfmncMeasures` in AcrossSessionResults docstring or a tiny `TrainTestPosErrMeasures.find_and_load_...` mirror only if load ergonomics need it; prefer the existing `extra_dfs` sink first.

## Out of scope

- Bapun train-test completion / clusterless paths
- Changing `perfmnc_session` semantics
- Notebook edits (unless you ask later)
- Publishing Figure 5 occupancy (separate from holdout position error)
