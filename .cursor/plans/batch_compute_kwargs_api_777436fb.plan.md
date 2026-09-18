---
name: batch compute kwargs API
overview: Refactor `batch_user_compute_fn` to accept a pipeline-style `computation_functions_name_includelist` + `computation_kwargs_dict`, with defaults inlined in the function (correctly mapped per computation name), then convert to the positional list `perform_specific_computation` requires.
todos:
  - id: batch-fn-api
    content: Extend batch_user_compute_fn with includelist + computation_kwargs_dict; inline correct default kwargs; merge overrides; convert to kwargs_list; replace broken WIP block
    status: completed
  - id: docstring-examples
    content: Document skip-heuristic and compute_2D override examples in batch_user_compute_fn docstring
    status: completed
isProject: false
---

# Extensible kwargs for `batch_user_compute_fn`

## Problem

[`batch_user_compute_fn`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/context_dependent.py) hardcodes which computations run and with which kwargs. The WIP `computation_kwargs_dict` around ~1141–1146 is also misaligned:

- `compute_2D` / `epochs_decoding_time_bin_size` belong on **`non_PBE_epochs_results`**, not `directional_decoders_evaluate_epochs`
- Heuristic scoring kwargs were assigned to the wrong key
- `perform_specific_computation` only accepts **`computation_kwargs_list`** (positional, same length as includelist) — see [`Computation.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py) ~1215–1244

## Approach

Match the existing pipeline convention (`resolve_and_execute` / `batch_extended_computations`): callers pass **name → kwargs**. Defaults stay **inline** in `batch_user_compute_fn` (no separate helper). Deep-merge user overrides, then convert to an aligned list for `perform_specific_computation`.

```mermaid
flowchart LR
  defaults["inline default includelist + kwargs_dict"] --> merge["merge user overrides"]
  merge --> list["aligned computation_kwargs_list"]
  list --> perform["perform_specific_computation"]
```

## Changes (single file)

### 1. Extend `batch_user_compute_fn` signature

```python
def batch_user_compute_fn(cls, curr_active_pipeline, force_recompute: bool = True, time_bin_size: float = 0.025, debug_print: bool = True,
                          computation_functions_name_includelist: Optional[List[str]] = None,
                          computation_kwargs_dict: Optional[Dict[str, Dict]] = None) -> GenericDecoderDictDecodedEpochsDictResult:
```

### 2. Inline defaults + merge (replace broken ~1141–1146)

Inside the function body:

**Default includelist** (same five as today, order preserved):

1. `merged_directional_placefields`
2. `directional_decoders_decode_continuous`
3. `directional_decoders_evaluate_epochs`
4. `directional_decoders_epoch_heuristic_scoring`
5. `non_PBE_epochs_results`

**Default kwargs** (correct mapping):

- `merged_directional_placefields`: `{ripple_decoding_time_bin_size, laps_decoding_time_bin_size}` ← `time_bin_size`
- `directional_decoders_decode_continuous`: `{time_bin_size: time_bin_size}`
- `directional_decoders_evaluate_epochs`: `{should_skip_radon_transform: True}`
- `directional_decoders_epoch_heuristic_scoring`: `{same_thresh_fraction_of_track: 0.05, max_ignore_bins: 2, use_bin_units_instead_of_realworld: False, max_jump_distance_cm: 60.0}`
- `non_PBE_epochs_results`: `{epochs_decoding_time_bin_size, frame_divide_bin_size: 10.0, compute_1D: True, compute_2D: False, drop_previous_result_and_compute_fresh: force_recompute, skip_training_test_split: True, debug_print_memory_breakdown: False}`

Merge logic:

1. Build `default_includelist` / `default_kwargs` inline as above
2. `includelist = computation_functions_name_includelist or default_includelist`
3. For each name in `includelist`: `merged[name] = {**default_kwargs.get(name, {}), **(computation_kwargs_dict or {}).get(name, {})}`
4. `computation_kwargs_list = [merged.get(name, {}) for name in includelist]`
5. Call `perform_specific_computation(..., computation_functions_name_includelist=includelist, computation_kwargs_list=computation_kwargs_list, ...)`

Default `compute_2D=False` (matches WIP / avoids the 14 GB OOM path). Callers that need 2D pass an override.

No `build_default_batch_user_compute_plan` helper.

### 3. Usage examples in docstring

```python
# skip heuristic scoring
batch_user_compute_fn(..., computation_functions_name_includelist=[
    'merged_directional_placefields', 'directional_decoders_decode_continuous',
    'directional_decoders_evaluate_epochs', 'non_PBE_epochs_results'])

# enable 2D non-PBE
batch_user_compute_fn(..., computation_kwargs_dict={'non_PBE_epochs_results': {'compute_2D': True}})
```

### 4. Caller compatibility

Existing call sites stay valid (new args optional):

- [`EpochComputationFunctions.perform_generalized_specific_epochs_decoding`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py) ~2181 — no change required
- Commented calls in `batch_user_completion_helpers.py` — unchanged

## Out of scope

- Display-path `force_recompute` default / Wave-1 size gating (separate fix)
- Accumulating multiple `time_bin_size`s in one generic result
- Changing `perform_specific_computation` to accept dicts natively
- Extracting a separate default-plan helper method
