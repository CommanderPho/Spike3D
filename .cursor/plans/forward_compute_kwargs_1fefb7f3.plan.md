---
name: forward compute kwargs
overview: Extend `perform_generalized_specific_epochs_decoding` to accept and forward `compute_2D` (default False), `computation_functions_name_includelist`, and `computation_kwargs_dict` into `batch_user_compute_fn`, and update the MultiColor display call sites to pass `compute_2D=False`.
todos:
  - id: extend-generalized-sig
    content: Add compute_2D / computation_functions_name_includelist / computation_kwargs_dict to perform_generalized_specific_epochs_decoding and forward to batch_user_compute_fn
    status: completed
  - id: update-display-kwargs
    content: Pass compute_2D=False from MultiColor display computation_kwargs_dict and fallback perform_specific_computation call
    status: completed
isProject: false
---

# Forward kwargs through `perform_generalized_specific_epochs_decoding`

## Context

[`batch_user_compute_fn`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py) already accepts `computation_functions_name_includelist` / `computation_kwargs_dict` and defaults `non_PBE_epochs_results.compute_2D=False`.

[`perform_generalized_specific_epochs_decoding`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py) (~2129) still calls it without those args (~2184), so pipeline/display kwargs cannot control nested Wave-2 behavior.

## Changes

### 1. Extend `perform_generalized_specific_epochs_decoding` signature

Add (single-line `def` if ≤400 chars, else grouped continuations):

- `compute_2D: bool = False`
- `computation_functions_name_includelist: Optional[List[str]] = None`
- `computation_kwargs_dict: Optional[Dict[str, Dict]] = None`

### 2. Forward into `batch_user_compute_fn` (~2184)

Before the call, merge the convenience flag into the nested kwargs (user dict wins on conflict):

```python
merged_computation_kwargs_dict: Dict[str, Dict] = dict(computation_kwargs_dict or {})
merged_computation_kwargs_dict['non_PBE_epochs_results'] = {
    'compute_2D': compute_2D,
    **(merged_computation_kwargs_dict.get('non_PBE_epochs_results') or {}),
}

a_new_fully_generic_result = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(
    curr_active_pipeline=owning_pipeline_reference,
    force_recompute=force_recompute,
    time_bin_size=epochs_decoding_time_bin_size,
    debug_print=debug_print,
    computation_functions_name_includelist=computation_functions_name_includelist,
    computation_kwargs_dict=merged_computation_kwargs_dict,
)
```

### 3. Update MultiColor display kwargs (~2819–2828 and ~2886–2887)

Pass `compute_2D=False` on `generalized_specific_epochs_decoding` in both `force_recompute` branches, and in the fallback `perform_specific_computation` kwargs list when generic result is missing:

```python
'generalized_specific_epochs_decoding': {
    'epochs_decoding_time_bin_size': time_bin_size,
    'drop_previous_result_and_compute_fresh': False,
    'force_recompute': ...,
    'compute_2D': False,
}
```

## Out of scope

- Changing `batch_user_compute_fn` defaults further
- Skipping heuristic scoring by default (callers can pass a custom includelist when desired)
- Fixing the unfinished “update existing generic result” path (~2188–2192)
