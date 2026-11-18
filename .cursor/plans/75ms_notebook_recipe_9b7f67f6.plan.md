---
name: 75ms notebook recipe
overview: "Point ProcessBatchOutputs_qclus1246789_Only.ipy at a post-rename 25ms→75ms continued run: reuse the existing qclu/frate local pickle, write a new -tbin_75ms global pickle/h5, and force-recompute only the tbin-dependent global functions."
todos:
  - id: force-recompute-list
    content: Set force_recompute_override_computations_includelist to the explicit tbin-dependent globals (include heuristic scoring + continuous decode; exclude rank-order).
    status: completed
  - id: narrow-include-list
    content: After get_run_configuration, override extended_computations_include_includelist so jonathan/rank-order/PF/FAT are not rebuilt into the 75ms global pickle.
    status: completed
  - id: sweep-075-only
    content: Set desired_shared_decoding_time_bin_sizes to [0.075] on the wcorr-shuffle sweep completion function.
    status: completed
  - id: recipe-comments
    content: Add a short post-rename 25ms→75ms comment block near ACTIVE PHASE documenting load/save and force-recompute contract.
    status: completed
isProject: false
---

# 25ms → 75ms ProcessBatchOutputs recipe

After the 25ms batch and [scripts/rename_session_outputs_add_tbin_25ms.ipy](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\scripts\rename_session_outputs_add_tbin_25ms.ipy) (`--execute`, mtime after 2026-10-19), 25ms session globals live at `-tbin_25ms`. The 75ms notebook should **not** overwrite those files.

Only [ProcessBatchOutputs_qclus1246789_Only.ipy](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy) changes. No pipeline/helper edits.

## Keep (already correct)

- `active_phase = ProcessingScriptPhases.final_run` (not `clean_run`).
- `override_custom_pickle_suffix` = `_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0` (load local pickle; no tbin).
- `laps_decoding_time_bin_size = 0.075`, `ripple_decoding_time_bin_size = 0.075` → `decoding_time_bin_filename_suffix` = `-tbin_75ms` (global pickle + h5 only).
- `drop_previous_result_and_compute_fresh=True` on tbin kwargs.
- `needs_recompute_heuristics=True` on `compute_and_export_decoders_saturated_posteriors_completion_function`.
- `override_custom_pickle_output_suffix` stays `None` so save uses the same qclu/frate stem + `-tbin_75ms` on global/h5.

Template paths (already wired):

- Load local: `{session}{override_custom_pickle_suffix}.pkl`
- Load/save global: `{session}{override_custom_pickle_suffix}-tbin_75ms.pkl`

25ms globals (`…-tbin_25ms.pkl`) are therefore left alone.

## Change: force-recompute list (main gap)

Today the notebook copies **all** `force_recompute_override_computation_kwargs_dict` keys, which includes `rank_order_shuffle_analysis` (not tbin work) and omits heuristic scoring / continuous decode.

Replace the derived list with an explicit tbin set:

```python
force_recompute_override_computations_includelist = [
    'merged_directional_placefields',
    'directional_decoders_decode_continuous',  # needed after 25→75ms placefields
    'directional_decoders_evaluate_epochs',
    'directional_decoders_epoch_heuristic_scoring',
    'non_PBE_epochs_results',
    'generalized_specific_epochs_decoding',
    'perform_wcorr_shuffle_analysis',
]
```

Keep `rank_order_shuffle_analysis` in the kwargs dict if you want those parameters around, but **do not** put it on the force-recompute list.

Keep `drop_previous_result_and_compute_fresh=True` on the tbin kwargs that already have it.

## Change: do not rebuild FAT / rank-order / PF / jonathan

A missing `-tbin_75ms` global pickle means `final_run`’s default phase-3 include list ([pythonScriptTemplating.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\pythonScriptTemplating.py) ~lines 70–86) would recompute jonathan, rank-order, PF peaks, etc. into the new 75ms global.

After `active_phase.get_run_configuration(...)`, override:

```python
extended_computations_include_includelist = [
    'pf_computation', 'split_to_directional_laps',  # local; validators no-op if already in loaded pickle
    'merged_directional_placefields',
    'directional_decoders_decode_continuous',
    'directional_decoders_evaluate_epochs',
    'directional_decoders_epoch_heuristic_scoring',
    'non_PBE_epochs_results',
    'generalized_specific_epochs_decoding',
    'perform_wcorr_shuffle_analysis',
]
```

Do **not** add `jonathan_firing_rate_analysis`, `rank_order_shuffle_analysis`, `long_short_decoding_analyses`, `long_short_fr_indicies_analyses`, `short_long_pf_overlap_analyses`, or FAT export functions as tbin work.

`custom_phase_extended_computations_include_includelist` can stay `None`; the post-config override is enough.

## Change: sweep bins (cheap extra)

`compute_and_export_session_alternative_replay_wcorr_shuffles_completion_function` currently uses `desired_shared_decoding_time_bin_sizes=[0.025, 0.060, 0.075, ripple]`. For this 75ms session-pickle run, set it to `[0.075]` so it does not re-decode 25ms/60ms. Collected_outputs names already carry `_tbin-0.075`; `final_run` may rewrite those CSVs.

## Comments to add near ACTIVE PHASE

Document the post-rename contract in the notebook:

- Do not set `clean_run`.
- Rename script must have moved 25ms globals to `-tbin_25ms` first.
- Load existing qclu/frate `loadedSessPickle`; write global pickle/h5 as `-tbin_75ms`.
- Force-recompute only the tbin-dependent globals above.

## Out of scope

- Empty-suffix refuse (separate request).
- Pipeline / template / parser / rename-script changes.
- Creating a 75ms global by copying 25ms then patching; load of `-tbin_75ms` is expected to miss and rebuild those globals.