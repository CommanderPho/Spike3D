---
name: Filter mask by decoder neuron_IDs
overview: Thread `included_neuron_ids` into `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` and pass `pfND_Decoder.neuron_IDs` from `_do_custom_decode_epochs` so bins with zero spikes from the active decoder's neurons are masked as inactive.
todos:
  - id: add-included-neuron-ids-param
    content: "Add `included_neuron_ids: Optional[NDArray]=None` to `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` in `reconstruction.py` and forward it to all three spike-count mask calls"
    status: completed
  - id: pass-decoder-neuron-ids-in-custom-decode
    content: Pass `included_neuron_ids=pfND_Decoder.neuron_IDs` in `_do_custom_decode_epochs` in `DirectionalPlacefieldGlobalComputationFunctions.py` and verify syntax/behavior
    status: completed
isProject: false
---

# Filter spike-count bin mask by decoder `neuron_IDs`

## Root Cause

In [`_do_custom_decode_epochs`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py), `global_spikes_df` contains the union of neurons across all directional decoders (`track_templates.any_decoder_neuron_IDs`), while `pfND_Decoder.decode_specific_epochs` only bins spikes from `pfND_Decoder.neuron_IDs`. Because `mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin` hardcodes `included_neuron_ids=None`, a test time bin with zero spikes from `pfND_Decoder.neuron_IDs` (but at least one spike from another direction/track's cell) is treated as active instead of masked to `NaN`, leaving the silence prior $\exp(-\tau \sum_c F_c(x))$ peak (at `xbin_centers[0]`) as a valid decoded position.

## Changes

1. **Add `included_neuron_ids` to `DecodedFilterEpochsResult.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin`** in [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) (lines 2496-2552):
   - Append `included_neuron_ids: Optional[NDArray]=None` to the single-line method signature (preserving existing behavior for callers that omit it, and remaining compatible with clusterless decoders where `neuron_IDs` is `None`).
   - Pass `included_neuron_ids=included_neuron_ids` to all three `spikes_df.spikes.compute_unit_time_binned_spike_counts_and_mask*` calls inside the `for i in np.arange(num_filter_epochs):` loop:
     - `compute_unit_time_binned_spike_counts_and_mask_from_window_intervals(..., included_neuron_ids=included_neuron_ids, ...)` (`sliding_iv is not None` branch, line 2537)
     - `compute_unit_time_binned_spike_counts_and_mask_from_window_intervals(..., included_neuron_ids=included_neuron_ids, ...)` (`a_time_bin_edges.ndim == 2` branch, line 2541)
     - `compute_unit_time_binned_spike_counts_and_mask(..., included_neuron_ids=included_neuron_ids, ...)` (`a_time_bin_edges.ndim == 1` branch, line 2543)

2. **Pass `included_neuron_ids=pfND_Decoder.neuron_IDs` in `_do_custom_decode_epochs`** in [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py) (lines 5701-5706):

```python
    if should_mask_by_required_spike_counts:
        decoder_result, _mask_index_tuple = decoder_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(global_spikes_df),
                                                                                                                                         min_num_spikes_per_bin_to_be_considered_active=min_num_spikes_per_bin_to_be_considered_active,
                                                                                                                                         min_num_unique_active_neurons_per_time_bin=min_num_unique_active_neurons_per_time_bin,
                                                                                                                                         masked_bin_fill_mode=masked_bin_fill_mode,
                                                                                                                                         included_neuron_ids=pfND_Decoder.neuron_IDs)
```

## Verification

- Run syntax and lint checks on [`reconstruction.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py) and [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py), plus a quick unit check verifying that a bin containing only non-decoder `aclu` spikes is masked (`is_time_bin_active == False`, `most_likely_positions == NaN`) when `included_neuron_ids=pfND_Decoder.neuron_IDs` is supplied.
