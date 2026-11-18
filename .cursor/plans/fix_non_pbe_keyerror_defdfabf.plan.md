---
name: Fix non_PBE KeyError
overview: Resolve the KeyError in `ComputationKWargParameters.init_from_pipeline` by mapping the merged function `short_name` `non_PBE_epochs_results` to the attrs field name `non_pbe_epochs_results`, without changing public computation strings or breaking saved pipelines.
todos:
  - id: alias-init-from-pipeline
    content: Add MERGED_FN_SHORT_NAME_TO_PARAM_FIELD map and use canonical_key in ComputationKWargParameters.init_from_pipeline loop (SpecificComputationParameterTypes.py).
    status: completed
  - id: verify-workflows
    content: Manually verify get_all_parameters / get_complete_session_identifier_string on pipeline with None computation_config; spot-check Bapun includelist cell.
    status: completed
  - id: optional-hdf
    content: (Optional) Add non_pbe_epochs_results (+ position_decoding if desired) to ComputationKWargParameters.to_hdf for parity with other fields.
    status: completed
isProject: false
---

# Fix `non_PBE_epochs_results` / `non_pbe_epochs_results` mismatch

## Root cause

- Merged computation functions are keyed by `[fn_best_name](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\function_helpers.py)` (decorator `short_name`), which for non-PBE epochs is `**non_PBE_epochs_results**` (`[EpochComputationFunctions.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py)` ~1905).
- `[ComputationKWargParameters.init_from_pipeline](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py)` looks up that string in `deepcopy(cls.__annotations__)`, where the field is named `**non_pbe_epochs_results**` (lowercase `pbe`) (~605).
- Python dict lookup is case-sensitive → `**KeyError: 'non_PBE_epochs_results'**`.
- This is **not format-specific**; it triggers whenever `global_computation_results.computation_config is None` and `[get_all_parameters](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py)` (~~3166–3170) calls `init_from_pipeline`—including export paths like `get_complete_session_identifier_string` → `get_custom_pipeline_filenames_from_parameters` (~~3402).

## Workflow compatibility (Bapun vs KDiba)

- **Bapun**: Interactive notebooks already use the **public** string `'non_PBE_epochs_results'` in `perform_specific_computation(..., includelist=[...])` and in `computation_kwargs_dict` keys (`[InteractivePipelineLoadFromPickle_Bapun_*.ipynb](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D)`). Those strings must stay as-is; they identify the registered merged function, not the attrs field.
- **KDiba / other formats**: Same pipeline code path; no KDiba-only branch in this logic. `[context_dependent.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\context_dependent.py)` and batch helpers also use `'non_PBE_epochs_results'`.
- **Repo search**: No Python code uses `all_params_dict['non_pbe_epochs_results']` or attribute `.non_pbe_epochs_results` outside the class definition; `get_custom_pipeline_filenames_from_parameters` only touches `rank_order_shuffle_analysis`, not this key. So fixing `init_from_pipeline` does not require notebook or batch string changes.

## Recommended implementation (minimal risk)

**Add a small alias map inside `ComputationKWargParameters.init_from_pipeline`** (same file as above), applied after `final_key = k.removeprefix('_')`:

- Map `**non_PBE_epochs_results` → `non_pbe_epochs_results**` (canonical attrs / constructor key).
- Lookup: `canonical_key = MERGED_FN_SHORT_NAME_TO_PARAM_FIELD.get(final_key, final_key)` then `a_type = params_class_type_dict[canonical_key]` and store `_out_param_typed_parameters_dict[canonical_key] = a_type(**v_dict)`.
- Keep a one-line comment explaining the mismatch (decorator `short_name` vs snake_case field).

**Why this over renaming the attrs field to `non_PBE_epochs_results`?**

- Renaming the field would align names but can **break unpickling** of existing saved `ComputationKWargParameters` / pipelines that embed the old field name in serialized state. The alias approach changes **only** the key-normalization step when building from registered functions and leaves stored objects and `to_dict()` keys as `**non_pbe_epochs_results**` (stable).

**Optional hygiene** (separate from the KeyError; only if you want completeness): `[ComputationKWargParameters.to_hdf](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\SpecificComputationParameterTypes.py)` (~720–736) currently omits `non_pbe_epochs_results` and `position_decoding`; adding them would match other fields. Not required to fix the reported traceback.

## Verification checklist (after implementing)

1. **Repro guard**: Load a pipeline with `computation_config is None`, call `get_all_parameters(allow_update_global_computation_config=False)` or `get_complete_session_identifier_string()`—should not raise.
2. **Bapun**: Run a cell that includes `perform_specific_computation(..., includelist=['non_PBE_epochs_results', ...])` (existing notebook pattern)—should still resolve the same registered function.
3. **KDiba**: Same as (1) on a KDiba pickle if available; no string changes expected.
4. **Regression**: Grep for any new merged functions whose `short_name` does not match `ComputationKWargParameters` annotation keys; extend the alias dict if the same pattern appears again (or add a unit test listing registered `short_name`s vs `__annotations`__ keys).

## Alternative (higher churn)

Rename `non_pbe_epochs_results` → `non_PBE_epochs_results` on `ComputationKWargParameters` and update any serialization/tests—only choose this if you explicitly accept **pickle migration** (or add a custom `__setstate_`_ / cattrs hook for old keys).