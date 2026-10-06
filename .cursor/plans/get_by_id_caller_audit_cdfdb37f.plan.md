---
name: get_by_id caller audit
overview: Every live BasePositionDecoder.get_by_id caller asks for a specific aclu list and needs those IDs kept. None of them depend on the old compute() path dropping cells below frate_thresh, so the new default should stay and no call site needs recompute=True.
todos: []
isProject: false
---

# get_by_id caller audit

No code changes. The new default (index the stored ratemap, `neuron_IDs` equal the requested ids) is what every live `BasePositionDecoder` caller is doing. The old `PfND.compute()` drop was an accidental second firing-rate cut.

`BayesianPlacemapPositionDecoder.get_by_id` and `PfND.get_by_id` are different methods. They still recompute. This change does not affect them.

## Callers that hit the new default

All of these pass an id list they already chose, and then require the copy to contain that list. Keeping every requested cell is the operation they describe.

- [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\DirectionalPlacefieldGlobalComputationFunctions.py) lines 652, 780, and 1421: union of the frate/qclu keep-set, then slice both long and short to that same list. This is the path that was raising `Assert.all_equal`.
- Same file, lines 1759, 1770, and 2044: `filtered_by_included_aclus` intersects the current ratemap ids with a qclu list, then slices.
- Same file, lines 1867, 2139, and 2451: build the shared-aclu one-step decoders from `BasePositionDecoder.init_from_stateful_decoder(...)`. The argument is the intersection of the two tracks' existing `neuron_IDs`. Those cells already survived the original placefield `frate_thresh`. The old recompute could drop them again and split long from short. Keeping the intersection is what this code asks for.
- Same file, lines 5902 and 7256: `TrainTestSplitResult.sliced_by_neuron_id` and the lap ground-truth cell subset. Both are typed as `BasePositionDecoder` and pass an explicit include list.
- [`reconstruction.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py) `prune_to_shared_aclus_only` (line 2849) and `perform_filter_by_frate` (line 3326). The frate method already selects `unsmoothed peak >= minimum_inclusion_fr_Hz` and then slices. The old `compute()` could drop more cells whose smoothed peak failed `pf.config.frate_thresh`. The method's own mask is the cut it documents.

One order change is intended. `Ratemap.get_by_id` used to return ratemap order. The new method returns the order of the `ids` argument. `perform_filter_by_frate` and `filtered_by_included_aclus` build `ids` with a mask on `neuron_ids`, so that order is already ratemap order. The shared-aclu builders pass one intersection array to both decoders, so both copies now share that array's order instead of each track's ratemap order. Pairwise equality still holds.

## Callers that do not use this method

These still run `PfND.compute()` inside their own `get_by_id`. Leave them as they are.

- `BayesianPlacemapPositionDecoder.get_by_id` ([`reconstruction.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction.py) line 3864), the DST override in [`reconstruction_dst.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\reconstruction_dst.py), [`tests/test_decoders.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_decoders.py), [`decoder_result.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\decoder_result.py) leave-one-out, [`eqn_debugger_export.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\eqn_debugger_export.py), and the Bayesian-annotated slices in [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py).
- `PfND.get_by_id` in [`neuropy/analyses/placefields.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\placefields.py) line 2102. That placefield helper has the same union-then-slice shape and can still drop cells. It is not on the decoder path this change fixed.
- Neuron, session, and `SpikeRateTrends` `get_by_id` calls are other types.

`PendingNotebookCode.py` line 3973 calls `dec.get_by_id` on the contextual pf2D decoders. Those objects are built by `build_contextual_pf2D_decoder`. If they are `BasePositionDecoder` instances, the new slice matches "restrict to `included_neuron_IDs`". If they are Bayesian decoders, they still use the override. Either way they are not asking for a second `frate_thresh` pass.

No call site should be switched to `recompute=True`.
