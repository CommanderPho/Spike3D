---
name: FAT per_epoch heuristics
overview: Merge HeuristicReplayScoring columns from DirectionalDecodersEpochsEvaluations onto FAT data_grain=='per_epoch' laps and PBE DFs after Phase 5, fault-tolerantly (log and continue). Heuristics already export separately as ripple_all_scores_merged_df CSVs — this adds them into the FAT per_epoch grain.
todos:
  - id: helper-merge-heuristics
    content: Add _helper_merge_heuristic_columns_onto_per_epoch_df using _build_merged_score_metric_df + start-time left merge
    status: completed
  - id: enrich-per-epoch-method
    content: Add adding_heuristic_score_columns_to_per_epoch_marginals with per-context try/except log-and-continue
    status: completed
  - id: wire-after-phase5
    content: Call enricher after Phase 5 in batch_user_compute_fn with DirectionalDecodersEpochsEvaluations if present
    status: completed
isProject: false
---

# Add heuristic columns to FAT per_epoch exports

## Existing export (answer)

Heuristics are **already exported**, but **not in FAT**:

- Separate CSV: `(ripple_all_scores_merged_df)_tbin-*.csv` via [`DecoderDecodedEpochsResult.export_csvs`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py) / batch `compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function`
- Built by `build_complete_all_scores_merged_df()` from **ripple** `filter_epochs` only (decoder-suffixed cols)
- **No** production `laps_all_scores_merged_df` writer (AcrossSessionResults only has a loader stub)
- FAT path (`default_export_all_CSVs`) exports marginal posteriors only — heuristics were never wired in

This plan embeds heuristic cols into FAT `per_epoch` so across-session FAT analysis does not need a separate join to `ripple_all_scores_merged_df`.

## Why after Phase 5

[`batch_user_compute_fn`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py) Phase 5 (`compute_all_per_epoch_aggregations_from_per_time_bin_results`, ~1364) **overwrites** `per_epoch` DFs from `per_time_bin` aggregations. Enriching only inside `adding_directional_pseudo2D...` would be lost. Enrich **once after Phase 5**.

```mermaid
flowchart LR
  eval[DirectionalDecodersEpochsEvaluations filter_epochs]
  merge[_build_merged_score_metric_df heuristic cols + suffixes]
  phase5[Phase 5 per_epoch DFs]
  fat[FAT CSV via default_export_all_CSVs]
  eval --> merge --> phase5 --> fat
```

## Changes (minimal)

### 1. Small helper to build + left-merge heuristic cols

File: [`context_dependent.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py)

Add a private/classmethod helper (near the x_meas / pseudo2D helpers), e.g. `_helper_merge_heuristic_columns_onto_per_epoch_df`:

- Inputs: `per_epoch_df`, `decoder_*_filter_epochs_decoder_result_dict` (laps or ripple), join key on epoch start
- Columns: `HeuristicReplayScoring.get_all_score_computation_col_names()` only (not full all_scores radon/wcorr/pearson)
- Build suffixed DF via existing `_build_merged_score_metric_df(..., best_decoder_index_column_name=None)`
- Join key: left DF `epoch_start_t` (fallback `lap_start_t` / `ripple_start_t`); right DF `start` (same as all_scores `ripple_start_t = start`)
- Left-merge; drop right-only join key; do not overwrite existing same-named cols
- Return enriched DF (or original on empty/missing cols)

### 2. Method on `GenericDecoderDictDecodedEpochsDictResult` that enriches all matching per_epoch contexts

Same file, e.g. `adding_heuristic_score_columns_to_per_epoch_marginals(self, directional_decoders_epochs_decode_result=None)`:

- If `directional_decoders_epochs_decode_result is None`: return self (no-op)
- Query contexts with `data_grain='per_epoch'` and `known_named_decoding_epochs_type in ('laps','pbe')`
- Map `laps` → `decoder_laps_filter_epochs_decoder_result_dict`, `pbe` → `decoder_ripple_filter_epochs_decoder_result_dict`
- For each matching context DF: call helper, write back into `filter_epochs_decoded_track_marginal_posterior_df_dict`
- **Fault tolerance (required):** wrap per-context and whole-method in `try/except` for `(AssertionError, ValueError, AttributeError, KeyError, TypeError)` — `print` a clear warning (same style as x_meas / most_likely_positions_1D) and continue; never abort export

### 3. Call after Phase 5 in `batch_user_compute_fn`

Same file ~1364–1365, after `compute_all_per_epoch_aggregations_from_per_time_bin_results`:

```python
directional_decoders_epochs_decode_result = curr_active_pipeline.global_computation_results.computed_data.get('DirectionalDecodersEpochsEvaluations', None)
a_new_fully_generic_result = a_new_fully_generic_result.adding_heuristic_score_columns_to_per_epoch_marginals(directional_decoders_epochs_decode_result=directional_decoders_epochs_decode_result)
```

If the global result key is missing, method no-ops with a short WARN log (same pattern as missing `DirectionalMergedDecoders`).

No export-path changes — columns ride into FAT via existing `default_export_all_CSVs`.

## Out of scope

- Changing / duplicating `(ripple_all_scores_merged_df)` export
- Adding radon/wcorr/pearson/best_dir columns (only heuristic computation col names)
- Notebook / across-session join fixes
- Re-export of historical FAT CSVs (needs recompute)
