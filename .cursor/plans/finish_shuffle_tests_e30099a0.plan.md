---
name: Finish shuffle tests
overview: The epoch-score shuffle machinery is already in SequenceBasedComputations.py. Remaining work is pipeline-free tests for multi-metric stats, pickle compat, heuristic scoring, and the batch reconstruct path—plus leaving existing decoder-indexing tests passing.
todos:
  - id: test-post-compute-metric
    content: Add post_compute_metric tests for abs_greater / greater / less and NaN exclusion
    status: completed
  - id: test-all-metrics-wide-df
    content: Construct DecodedEpochScoreShuffle from arrays; assert post_compute_all_metrics + wide DF z/p columns
    status: completed
  - id: test-pickle-compat
    content: Cover __setstate__ old-field fill, 3-tuple and 7-tuple load_and_append_shuffle_data, get_dict_subset reconstruct
    status: completed
  - id: test-heuristic-mock
    content: Mock decode + templates; assert compute_shuffle_heuristic_score_arrays shapes and no filter_epochs mutation
    status: completed
  - id: run-unittest
    content: Run decoder + new epoch-score shuffle unittests
    status: completed
isProject: false
---

# Finish epoch-score shuffle tests

The rename, shuffle loop, lightweight heuristics, radon extract, `post_compute_all_metrics`, wide-DF z/p columns, dual CSV keys, and 7-tuple pickle load are already in [`SequenceBasedComputations.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/SequenceBasedComputations.py). Do **not** re-implement those.

This plan only closes the open `pipeline-batch-tests` todo: add unit tests that do not need a live pipeline.

## Out of scope

- Batch includelists stay on `wcorr_shuffle_analysis` (wrapper is registered).
- Notebooks, `_shuffle_pf1D_decoder` semantics, merging into `DirectionalDecodersEpochsEvaluations`.
- Rewriting [`test_pickling_results.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/tests/test_pickling_results.py) (it still requires `curr_active_pipeline`).

## File

Add [`pyPhoPlaceCellAnalysis/tests/test_epoch_score_shuffle.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/tests/test_epoch_score_shuffle.py) next to the existing decoder test. Keep [`test_wcorr_shuffle_decoder.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/tests/test_wcorr_shuffle_decoder.py) unchanged.

Follow existing test import bootstrap (`neuropy.core.position` first, then src path). Use `DecodedEpochScoreShuffle` (alias `WCorrShuffle` is fine). Two blank lines between methods; `## END for ...` on any new loops.

## 1. `post_compute_metric` extreme modes

Call `DecodedEpochScoreShuffle.post_compute_metric` on synthetic `(n_shuffles, n_epochs, n_decoders)` stacks. No object construction needed.

- `abs_greater` (wcorr): real `|r|` vs shuffle `|s|`; count more-extreme, p, z.
- `greater` (coverage): shuffle `>` real.
- `less` (travel / avg_jump_cm): shuffle `<` real.
- NaNs in the shuffle stack are excluded from `n_valid` (current `np.logical_not(np.isnan(...))`).

## 2. `post_compute_all_metrics` + wide DF columns

Construct a `DecodedEpochScoreShuffle` with only arrays (no pipeline):

- `filtered_epochs_df` with `start`/`stop`/`duration` plus `long_best_dir_decoder_IDX` / `short_best_dir_decoder_IDX`
- `real_decoder_ripple_weighted_corr_arr` and `real_score_arrays` including `wcorr` plus one `greater` metric (`coverage`) and one `less` metric (`travel`)
- `output_extracted_score_arrays_list` with matching per-shuffle dicts

Assert `post_compute_all_metrics` returns those three keys with shapes `(n_epochs, 4)`.

Call `build_all_shuffles_dataframes(decoder_names=DEFAULT_SHUFFLE_DECODER_NAMES)` and assert wide columns exist: existing `wcorr_long_LR` / `wcorr_z_long`, extra `{coverage,travel}_{decoder}`, `{coverage,travel}_z_{decoder}`, `{coverage,travel}_p_{decoder}`, `{coverage,travel}_n_extreme_{decoder}`. Keep `post_compute`’s 4-tuple return unchanged (compat).

## 3. Pickle `__setstate__` and 3/5/7-tuple payloads

- Old-state `__setstate__`: pass a dict that has `output_extracted_result_wcorrs_list` + `real_decoder_ripple_weighted_corr_arr` but **omits** `real_score_arrays` / `output_extracted_score_arrays_list` / flags. After `__setstate__`, `real_score_arrays['wcorr']` is filled, score-array list is synthesized from the DF dicts, flags default True/False/False.
- Tuple load via `load_and_append_shuffle_data`: write a temp pickle with `saveData` as a 3-tuple and a 7-tuple; start from an object whose `real_decoder_ripple_weighted_corr_arr` matches (required by the `allclose` guard). 7-tuple must copy `output_extracted_score_arrays_list` and `real_score_arrays`. 3-tuple must synthesize wcorr-only score arrays.
- Do **not** call `init_from_data_only_file` (it needs a pipeline).

## 4. Batch reconstruct path

Mirror [`finalize_output_shuffled_epoch_scores`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_completion_helpers.py) L768:

```python
WCorrShuffle(**get_dict_subset(obj.to_dict(), subset_excludelist=['_VersionedResultMixin_version']))
```

Assert the rebuilt object keeps `real_score_arrays`, `enable_heuristic_shuffle_scoring`, and `n_completed_shuffles`.

## 5. Lightweight heuristic API (mocked decode, no pipeline)

Duck-typed mocks only (do not construct full `TrackTemplates` / `DecodedFilterEpochsResult`):

- Result: `num_filter_epochs`, `most_likely_positions_list`, `p_x_given_n_list`, `nbins`, `time_window_centers`, `time_bin_edges`, `filter_epochs` (a DataFrame you can identity-check).
- Templates: `long_LR_decoder.xbin` and `get_track_length_dict()`.

Call `HeuristicReplayScoring.compute_shuffle_heuristic_score_arrays`. Assert:

- return dict values are `(n_epochs, n_decoders)`
- expected keys include MAP-only `travel` / `coverage` / `avg_jump_cm` and partition keys (`mseq_len`, `forwardicity`, …)
- `filter_epochs` is the same object / equal to the pre-call copy (no mutation)
- does **not** require `compute_pho_heuristic_replay_scores`

Use ≥2 time bins so partition/MAP scores have a defined jump.

## 6. Run

`uv run python -m unittest pyphoplacecellanalysis.tests.test_wcorr_shuffle_decoder pyphoplacecellanalysis.tests.test_epoch_score_shuffle` from the pyPhoPlaceCellAnalysis package (or the equivalent path the existing decoder test uses). Both must pass.
