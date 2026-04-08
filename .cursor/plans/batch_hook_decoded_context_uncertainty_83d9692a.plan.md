---
name: Batch hook decoded context uncertainty
overview: Hook `determine_decoded_context_uncertainty_as_fn_of_position` into the batch pipeline via a standard user completion function, registered in the same place as other decoding exports, and ordered after `EpochComputations` / generalized decode is guaranteed.
todos:
  - id: add-wrapper
    content: Add export_decoded_context_uncertainty_by_position_completion_function in batch_user_completion_helpers.py (call PendingNotebookCode + collected_outputs_path, batch-safe flags)
    status: completed
  - id: register-dict
    content: Register wrapper after generalized_decode in MAIN_get_template_string default dict and/or pythonScriptTemplating phase3 (match ProcessBatchOutputs.ipy path)
    status: completed
  - id: imports-scripts
    content: Update pythonScriptTemplating imports if needed; regenerate batch scripts from ProcessBatchOutputs
    status: completed
isProject: false
---

# Add decoded context–by–position to batch user outputs

## How batch “user outputs” run today

Per-session work finishes in [`BatchCompletionHandler.py`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py): after global computations, `on_complete_success_execution_session` runs `self.completion_functions` **in list order** (see ~770–785). Each callable has the usual signature `(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict, **kwargs)` and should return the updated `across_session_results_extended_dict`.

Your target function in [`PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) only needs `curr_active_pipeline` plus export/display kwargs. It reads `curr_active_pipeline.global_computation_results.computed_data['EpochComputations']` and `a_generic_decoder_dict_decoded_epochs_dict_result`, so it must run **after** that data exists (typically after [`generalized_decode_epochs_dict_and_export_results_completion_function`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) in the same completion sequence).

## Where to register (pick the path that matches how you generate scripts)

[`ProcessBatchOutputs.ipy`](h:/TEMP/Spike3DEnv_KDibaVersion/Spike3D/ProcessBatchOutputs.ipy) currently sets `phase_any_run_custom_user_completion_functions_dict = None` and passes that to `MAIN_get_template_string` (lines ~115–223). When the override is `None`, **`MAIN_get_template_string` uses its own large default dict** in [`batch_user_completion_helpers.py`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) (~4003–4027), **not** `ProcessingScriptPhases.get_custom_user_completion_functions_dict()`.

So:

| How you build batch scripts | Where to add the new completion function |
|-----------------------------|------------------------------------------|
| **`phase_any_run_custom_user_completion_functions_dict` is `None`** (current default in `ProcessBatchOutputs.ipy`) | Insert the new entry in **`MAIN_get_template_string`’s default `custom_user_completion_functions_dict`**, **immediately after** `'generalized_decode_epochs_dict_and_export_results_completion_function'` and **before** `'figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'`, so decode data is finalized before your analysis and before optional figure work. |
| You **uncomment** `custom_user_completion_functions_dict = active_phase.get_custom_user_completion_functions_dict(...)` (line ~218) | Add the same entry to **`phase3_any_run_custom_user_completion_functions_dict`** in [`pythonScriptTemplating.py`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py) (~184–197) in the same relative position (after `generalized_decode_...`, before `figures_plot_generalized_decode_...`). You can also use `extra_run_functions={...}` to merge a single new function without editing the phase dict. |
| Custom **override dict** in `ProcessBatchOutputs.ipy` | Include the new function in that dict with the **same ordering** relative to `generalized_decode_epochs_dict_and_export_results_completion_function`. |

**Phase note:** `final_run` uses phase3 extended computations (includes `generalized_specific_epochs_decoding`, etc.). If you only add the hook to phase3 / `MAIN_get_template_string` defaults, **clean_run / continued_run** jobs will not run it unless you also add it to the corresponding phase dict or your override.

## Implementation pattern (recommended)

1. **Add a thin wrapper** in [`batch_user_completion_helpers.py`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) (same file as `generalized_decode_epochs_dict_and_export_results_completion_function`), e.g. `export_decoded_context_uncertainty_by_position_completion_function`, that:
   - `assert self.collected_outputs_path.exists()` (same pattern as `compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function`).
   - Lazy-imports `determine_decoded_context_uncertainty_as_fn_of_position` from `PendingNotebookCode` (this file already imports from `PendingNotebookCode` elsewhere—e.g. ~1063).
   - Calls `determine_decoded_context_uncertainty_as_fn_of_position(curr_active_pipeline, time_bin_size=..., enable_export_path=self.collected_outputs_path.resolve(), show_pos_by_ctxt_joint_figure=False)` so batch runs stay headless and write under `collected_outputs/output/` as your function already does.
   - Wraps the call in try/except if you want parity with other completions that must not kill the whole session on failure; your CSV/PNG paths already catch per-file errors.
   - Stores a small summary in `across_session_results_extended_dict['export_decoded_context_uncertainty_by_position_completion_function']` (e.g. partition keys or `None` on failure) for later inspection from `PipelineCompletionResult.across_session_results`.

2. **Import the new wrapper** in [`pythonScriptTemplating.py`](h:/TEMP/Spike3DEnv_KDibaVersion/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py) next to the other `batch_user_completion_helpers` imports if you register it in the phase3 dict.

3. **Regenerate** batch Python/Slurm scripts from `ProcessBatchOutputs*.ipy` after changing registrations so the generated scripts `append` the new function in the right order.

## Optional: `time_bin_size` vs batch decode

In `determine_decoded_context_uncertainty_as_fn_of_position`, `time_bin_size` is mainly used for **export filenames / metadata**, not for selecting the decoder context. For consistent naming with your batch decode settings, pass the same value via `BatchSessionCompletionHandler.override_user_completion_function_kwargs_dict` keyed by the new function’s `__name__`, or add a parameter on the wrapper defaulting to `0.060` / your batch `laps_decoding_time_bin_size`.

## Summary

**Safest placement:** a new `*_completion_function` in `batch_user_completion_helpers.py`, registered **immediately after** `generalized_decode_epochs_dict_and_export_results_completion_function` in whichever dict actually feeds your script generation (`MAIN_get_template_string` defaults if `ProcessBatchOutputs.ipy` keeps `None`, else `pythonScriptTemplating` phase3 or your override). Use `enable_export_path=self.collected_outputs_path` and `show_pos_by_ctxt_joint_figure=False` for supercomputer runs.
