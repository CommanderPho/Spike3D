---
name: WCorr shuffle IndexError
overview: The IndexError is a neuron-index mismatch inside the wcorr shuffle, not a stale shuffle file. Recomputing the session will not fix it; the template decoder’s ratemap is longer than its `neuron_IDs`, and the shuffle indexes the short array with positions from the long one.
todos:
  - id: confirm-lengths
    content: Print len(neuron_IDs) vs len(pf.ratemap.neuron_ids) for each template decoder to confirm the 33 vs >=42 mismatch
    status: completed
  - id: fix-shuffle-index
    content: After confirmation, fix _shuffle_pf1D_decoder so ratemap, neuron_IDs, and F are permuted by one index vector into the array being indexed
    status: completed
isProject: false
---

# WCorr shuffle IndexError

Recomputing the session will not fix this. The crash is in the shuffle code, on one of the four track-template decoders, and the same failure is already marked on that line from 2025-01-20 (`index 21` vs size `16`; this run is `index 41` vs size `33`).

## What failed

`compute_and_export_session_wcorr_shuffles_completion_function` calls `WCorrShuffle.compute_shuffles(num_shuffles=2)` before any previously saved shuffles are loaded. The merged decoder shuffle at line 686 of [`SequenceBasedComputations.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\SequenceBasedComputations.py) succeeds. The dict comprehension on the next line fails while shuffling `track_templates.get_decoders_dict()`.

Inside [`WCorrShuffle._shuffle_pf1D_decoder`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\SequenceBasedComputations.py):

```387:402:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\SequenceBasedComputations.py
    def _shuffle_pf1D_decoder(cls, a_pf1D_Decoder: BasePositionDecoder, shuffle_IDXs: NDArray, shuffle_aclus: NDArray) -> BasePositionDecoder:
        a_shuffled_decoder = deepcopy(a_pf1D_Decoder)
        is_shuffle_aclu_included = np.isin(shuffle_aclus, a_shuffled_decoder.pf.ratemap.neuron_ids)
        shuffle_aclus = shuffle_aclus[is_shuffle_aclu_included]
        shuffle_IDXs = [list(a_shuffled_decoder.pf.ratemap.neuron_ids).index(aclu) for aclu in shuffle_aclus]
        a_shuffled_decoder.pf.ratemap = a_shuffled_decoder.pf.ratemap.get_by_id(shuffle_aclus)
        neuron_indexed_field_names = ['neuron_IDs', 'neuron_IDs']
        for a_field in neuron_indexed_field_names:
            setattr(a_shuffled_decoder, a_field, getattr(a_shuffled_decoder, a_field)[shuffle_IDXs])
```

`shuffle_IDXs` here are positions in `pf.ratemap.neuron_ids`. Index `41` means that ratemap has at least 42 neurons. Those positions are then used on `decoder.neuron_IDs`, which has length 33.

The `shuffle_IDXs` argument from `build_shuffled_ids` (indices into the merged decoder) is discarded and recomputed. This is not “global permutation indices applied to a smaller decoder.”

## Why the two arrays differ

`BasePositionDecoder.setup` / `post_load` set `neuron_IDs` from `pf.ratemap.neuron_ids` via `build_concatenated_F`, so a freshly set-up decoder has equal lengths. The merged decoder is built that way (`BasePositionDecoder(..., setup_on_init=True)`), which is why its shuffle passes.

The four template decoders are not guaranteed to stay that way. `decoder_neuron_IDs_list` is `pf.ratemap.neuron_ids`, so the shuffle walks the full ratemap. `neuron_IDs` on that same object is the shorter array. A full recompute of `DirectionalLaps` / `DirectionalMergedDecoders` only helps if the new objects happen to come out of `setup()` still aligned. The shuffle function still assumes that alignment and will crash again on the next decoder where the ratemap is longer.

Previous saved shuffles are not involved. `discover_load_and_append_shuffle_data_from_directory` runs after `compute_shuffles`.

## Check before any code change

On the live template decoders, print both lengths. The failing decoder is the first one where the ratemap is longer:

```python
for name, dec in track_templates.get_decoders_dict().items():
    print(name, 'neuron_IDs', len(dec.neuron_IDs), 'ratemap', len(dec.pf.ratemap.neuron_ids))
```

If they differ, `dec.setup()` rebuilds `neuron_IDs` and `F` from the current ratemap. That clears this IndexError only while the arrays stay the same length. It does not make the shuffle correct.

## Why a length-matched rerun would still be wrong

Even when the lengths match, this function does not permute the tuning curves:

- [`Ratemap.get_by_id`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\ratemap.py) uses `np.isin`, a boolean mask in the original neuron order. It does not reorder to `shuffle_aclus`.
- `a_shuffled_decoder.F[shuffle_IDXs, :] = a_shuffled_decoder.F[shuffle_IDXs, :]` assigns the array to itself.
- Only `neuron_IDs` is reordered, and only when it is the same length and order as the ratemap.

A later code fix has to permute the ratemap, `neuron_IDs`, and `F` with one index vector taken from the array actually being indexed. No session recompute is required first.
