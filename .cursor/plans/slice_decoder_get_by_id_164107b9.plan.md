---
name: Slice decoder get_by_id
overview: Make BasePositionDecoder.get_by_id return exactly the requested neuron IDs by indexing the existing ratemap. Recomputing, which reapplies the placefield firing-rate cut and drops different cells on long versus short, becomes an explicit flag.
todos:
  - id: decoder-get-by-id
    content: Add recompute=False to BasePositionDecoder.get_by_id and index the existing ratemap, spikes, and reliability arrays in requested-id order
    status: completed
  - id: revert-call-site
    content: Remove the broken inline slice loop in TrackTemplates.determine_decoder_aclus_filtered_by_frate_and_qclu and keep a single get_by_id call
    status: completed
isProject: false
---

# Slice decoder neuron IDs without recomputing

`BasePositionDecoder.get_by_id` in [`reconstruction.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py) currently calls `PfND.get_by_id`, and that always runs `compute()`. `compute()` keeps only cells whose smoothed peak is above `pf.config.frate_thresh`, so long and short decoders given the same aclu list come back with different `neuron_IDs`. `TrackTemplates.determine_decoder_aclus_filtered_by_frate_and_qclu` then fails `Assert.all_equal`.

Default behavior becomes the docstring contract: the copy's `neuron_IDs` equal the requested `ids`, in that order, taken from the tuning curves already stored. `recompute=False` is the default. `recompute=True` keeps today's `PfND.get_by_id` path, which reapplies `pf.config.frate_thresh` and may return a shorter list. That flag does not apply the rank-order `minimum_inclusion_fr_Hz` or qclu cuts; those stay in the template filter that builds the id list.

```python
def get_by_id(self, ids, defer_compute_all:bool=False, recompute:bool=False):
```

`defer_compute_all` stays unused on this class, as it is now.

## Default path (`recompute=False`)

- Deepcopy `self.pf`. Require every requested id to already be in `ratemap.neuron_ids` (same rule as [`Ratemap.get_by_id`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\ratemap.py)).
- Index the ratemap in requested-id order, not `np.isin` order, so `neuron_ids` matches `ids` exactly. `np.union1d` sorts the shared list, and the later `filtered_direction_shared_aclus_list[0] == shared_LR_aclus_only_neuron_IDs` check compares that order.
- Restrict `_filtered_spikes_df` with `np.isin` on `aclu`. Slice `_ratemap_spiketrains` and `_ratemap_spiketrains_pos` with the same index vector when they are present. Do not call `compute()`.
- Build `BasePositionDecoder(sliced_pf, setup_on_init=True, post_load_on_init=False, ...)`. `setup()` rebuilds `neuron_IDs` and `F` from the sliced ratemap.
- Copy `should_discount_silence`, `drop_negative_contributing_terms_mode`, and `reliability_modifier_mode`. Slice `reliability_active` and `reliability_silent` with the same integer index vector via `_slice_reliability_array` so their neuron axis matches `neuron_IDs`.

## Opt-in path (`recompute=True`)

Keep the current body: `self.pf.get_by_id(ids)` then wrap it in a new `BasePositionDecoder`. Document that `neuron_IDs` can be a subset of `ids`.

## Call site

[`DirectionalPlacefieldGlobalComputationFunctions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) around line 1419 still executes `a_decoder.get_by_id(...)` and then a second loop that references an undefined `ids`. Remove that loop and leave one `get_by_id` call. `BaseTrackTemplates` at line 780 already does that and needs no local slice.

`BayesianPlacemapPositionDecoder.get_by_id` is a separate override used by [`tests/test_decoders.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_decoders.py). Leave it unchanged. `PfND.get_by_id` stays the recompute implementation.