---
name: Sync parameter classes
overview: Most parameter classes in SpecificComputationParameterTypes.py still match their computation-function kwargs. Three classes are out of date. Rank-order defaults should stay as they are. Do not regenerate the file from the templating helper.
todos:
  - id: add-slideby
    content: Add slideby to directional_decoders_decode_continuous_Parameters and types_override_dict
    status: completed
  - id: sync-non-pbe
    content: Set non_PBE epochs_decoding_time_bin_size to 0.05 and add IGNORE_MEMORY_ERROR_FOR_DEBUGGING
    status: completed
  - id: add-force-recompute
    content: Add force_recompute to perform_specific_epochs_decoding_Parameters
    status: completed
isProject: false
---

# Sync drifted computation parameter classes

The classes in [`SpecificComputationParameterTypes.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py) were generated on 2024-10-07 from each registered function's default kwargs (`GlobalComputationParametersAttrsClassTemplating`, ignoring `include_includelist` and `debug_print`). I compared every `*_Parameters` class to the current `short_name` function signature.

## Leave these alone

These match the current signatures (names and defaults):

- `merged_directional_placefields_Parameters`
- `directional_decoders_evaluate_epochs_Parameters`
- `directional_decoders_epoch_heuristic_scoring_Parameters`
- `long_short_decoding_analyses_Parameters`
- `long_short_rate_remapping_Parameters`
- `long_short_inst_spike_rate_groups_Parameters`
- `wcorr_shuffle_analysis_Parameters`
- `position_decoding_Parameters`
- `DEP_ratemap_peaks_Parameters`
- `ratemap_peaks_prominence2d_Parameters`

`rank_order_shuffle_analysis_Parameters` still has the same four fields (`num_shuffles`, `minimum_inclusion_fr_Hz`, `included_qclu_values`, `skip_laps`). The function now defaults all of them to `None` so a call can mean "use the config." The class still holds the real defaults (`500`, `2.0`, `[1, 2, 4, 6, 7, 8, 9]`, `False`). [`ComputationKWargParameters.init_from_pipeline`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py) already drops `None` before constructing the object so those class defaults are not wiped. Do not copy the function's `None` defaults onto the class.

`directional_train_test_split_Parameters.training_data_portion` is `0.8333333333333334`; the function writes `5.0/6.0`. Same value.

## Update these three

`init_from_pipeline` builds each class with `a_type(**function_defaults)`, after dropping `None`. A new non-`None` kwarg that the class does not declare raises `TypeError`.

- [`directional_decoders_decode_continuous_Parameters`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py): `_decode_continuous_using_directional_decoders` gained `slideby: Optional[float] = None` ([DirectionalPlacefieldGlobalComputationFunctions.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) around line 8473). `None` is stripped today, so init does not crash, but the config cannot store a slide step. Add `slideby` plus `slideby_PARAM` (`param.Number`, default `None`), next to `time_bin_size`.
- [`non_PBE_epochs_results_Parameters`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py): `perform_compute_non_PBE_epochs` now defaults `epochs_decoding_time_bin_size` to `0.050` (class still says `0.025`) and has `IGNORE_MEMORY_ERROR_FOR_DEBUGGING: bool = False`. The boolean is not `None`, so `init_from_pipeline` will `TypeError`. Change the bin-size default (attrs field and `param.Number`) to `0.05`, and add the boolean field plus its `param.Boolean`.
- [`perform_specific_epochs_decoding_Parameters`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py): `_perform_specific_epochs_decoding` gained `force_recompute: bool = False`. Same `TypeError` on init. Add the field and `param.Boolean`.

Also add `slideby` to `types_override_dict` in [`PipelineParameterClassTemplating.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\PipelineParameterClassTemplating.py) so a later manual regen keeps `Optional[float]` instead of inferring `NoneType`.

Do not run `main_generate_params_classes`. It would rewrite every class, turn the rank-order defaults into `None`, and emit a class for every registered function that has kwargs, including display helpers.

## Not in this edit

Several newer computation functions have default kwargs and no parameter class at all (`generalized_specific_epochs_decoding`, `predictive_decoding_analysis`, the clusterless / two-step position decoders, `temporal_sequentiality_measures`). Display methods on the same registry (`directional_laps_overview`, and the other `_display_*` short names) also have kwargs. `init_from_pipeline` looks up every registered short name on `ComputationKWargParameters` and raises on the first miss. That is a separate gap from the three classes above.
