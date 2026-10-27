---
name: Rename shuffle completion fn
overview: Rename `compute_and_export_session_wcorr_shuffles_completion_function` to `compute_and_export_session_epoch_score_shuffles_completion_function`, switch its internals to `DecodedEpochScoreShuffle` / `ripple_score_shuffle` (no aliases), and update generator registrations with no leftover wrapper.
todos:
  - id: rewrite-completion-fn
    content: Rename completion function; switch internals to DecodedEpochScoreShuffle / ripple_score_shuffle; record all export_csvs keys
    status: completed
  - id: update-template-string
    content: Update MAIN_get_template_string and pythonScriptTemplating import/commented dict to the new function name
    status: completed
  - id: update-generator-ipy
    content: Update ProcessBatchOutputs*.ipy imports, phase completion dicts, and override kwargs keys; no alias
    status: completed
isProject: false
---

# Rename epoch-score shuffle completion function

No alias. New name: `compute_and_export_session_epoch_score_shuffles_completion_function`.

Do not edit historical `.ipynb` files (`ReviewOfWork_*`, `_DEP_OldNotebooks`, `PostBatchCopyResults.ipynb`). Those will keep importing the old name until edited separately.

Leave `export_csvs` dual CSV keys (`ripple_WCorrShuffle_df` + `ripple_EpochScoreShuffle_df`) as they are; those are on-disk identifiers, not Python aliases.

## 1. Rewrite the completion function

In [`batch_user_completion_helpers.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py):

- Rename `def compute_and_export_session_wcorr_shuffles_completion_function` → `compute_and_export_session_epoch_score_shuffles_completion_function`.
- Import `DecodedEpochScoreShuffle` (not `WCorrShuffle`).
- Access the container via `.ripple_score_shuffle` get/set. Attrs construction still uses the serialized field: `SequenceBasedComputationsContainer(wcorr_ripple_shuffle=None, is_global=True)`.
- Reconstruct with `DecodedEpochScoreShuffle(**get_dict_subset(...))`.
- Local variable: `epoch_score_shuffles`.
- Write callback results under the **new** dict key.
- Record all `export_csvs` paths that exist (`ripple_WCorrShuffle_df`, `ripple_EpochScoreShuffle_df`, `ripple_EpochScoreShuffle_long_df`).
- New standalone pickle stem: `standalone_epoch_score_shuffle_data_only` (discover already accepts that glob). Keep writing `.mat` via `save_data_mat`; filename can stay wcorr-array because that payload is still the wcorr stack.

Also update `MAIN_get_template_string` (same file, ~L5350): dict key + value → the new function name.

## 2. Generator path registrations

Replace the import, phase-dict key, and kwargs-dict key in:

- [`pythonScriptTemplating.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py) (import + commented default dict)
- [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_qclus1246789_Only.ipy) (import, `final_run` completion dict, `custom_user_completion_function_override_kwargs_dict`)
- Sibling generators with the same symbols: `ProcessBatchOutputs.ipy`, `ProcessBatchOutputs_qclus12_Only.ipy`, `ProcessBatchOutputs_test_all_script_phases.ipy`, `ProcessBatchOutputs_Bapun_Batch.ipy`, `ProcessBatchOutputs_NWB_WMaze_Batch.ipy`

Do not add a `compute_and_export_session_wcorr_shuffles_completion_function = ...` alias.

## Out of scope

- `compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function`
- `WCorrShuffle` / `wcorr_ripple_shuffle` aliases on the result class (pickle compat)
- Mass notebook edits
