---
name: Include laps in posteriors H5
overview: The HDF5 saver already writes both laps and ripples when both dicts are passed. The ripple-only “default” comes from call sites that pass `decoder_laps_filter_epochs_decoder_result_dict=None`; update those to pass the laps dict like `save_custom` already does.
todos:
  - id: fix-compute-export-call
    content: Pass decoder_laps_filter_epochs_decoder_result_dict (deepcopy) instead of None in compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function
    status: completed
  - id: fix-docstring-example
    content: Update PosteriorExporting class docstring example to pass both laps and ripple dicts
    status: completed
isProject: false
---

# Include lap posteriors in default `(decoded_posteriors)` H5

## How it works today

[`PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/data_exporting.py) already saves **both** epoch types when non-`None`:

```937:941:H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/data_exporting.py
out_contexts = {'laps': None, 'ripple': None}
if decoder_laps_filter_epochs_decoder_result_dict is not None:
    out_contexts['laps'] = _subfn_perform_save_single_epochs(..., epochs_name='laps', ...)
if decoder_ripple_filter_epochs_decoder_result_dict is not None:
    out_contexts['ripple'] = _subfn_perform_save_single_epochs(..., epochs_name='ripple', ...)
```

HDF keys get `epochs_name=laps` or `epochs_name=ripple` in the context path.

Two call paths diverge:

| Call site | Laps? | Ripples? |
|-----------|-------|----------|
| [`save_custom_session_files_completion_function`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) ~2901–2904 | Yes (passes both dicts) | Yes |
| [`compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) ~997–999 | **No** (`=None`) | Yes |
| Class docstring example in [`data_exporting.py`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/data_exporting.py) ~207–209 | **No** (`=None`) | Yes |

If you call the API yourself, pass both:

```python
out_contexts, paths = PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5(
    decoder_laps_filter_epochs_decoder_result_dict=deepcopy(directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict),
    decoder_ripple_filter_epochs_decoder_result_dict=deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict),
    _save_context=_parent_save_context, save_path=save_path,
)
```

Expect log lines for both `*_laps` and `*_ripple` (as in your earlier `save_custom` run).

When loading, do not slice to ripple-only (the helper example currently does `_out_ripple_only_dict = {k: v['ripple'] for k, v in _out_dict.items()}`). Keep `v['laps']` as well.

## Code change (default batch export)

In [`batch_user_completion_helpers.py`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py) inside `compute_and_export_decoders_epochs_decoding_and_evaluation_dfs_completion_function` (~997), replace:

```python
decoder_laps_filter_epochs_decoder_result_dict=None,
decoder_ripple_filter_epochs_decoder_result_dict=deepcopy(...decoder_ripple...),
```

with:

```python
decoder_laps_filter_epochs_decoder_result_dict=deepcopy(directional_decoders_epochs_decode_result.decoder_laps_filter_epochs_decoder_result_dict),
decoder_ripple_filter_epochs_decoder_result_dict=deepcopy(directional_decoders_epochs_decode_result.decoder_ripple_filter_epochs_decoder_result_dict),
```

Mirror the same change in the [`PosteriorExporting`](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/data_exporting.py) class docstring usage example (~207) so docs match behavior.

Leave `save_custom` as-is (already correct). Leave the specialized PBE/replay-only calls ~3706/3711 alone (they intentionally pass one side as `None`).

## Verify

Re-export and confirm HDF groups / context keys include both `epochs_name=laps` and `epochs_name=ripple` for each decoder (`long_LR`, `long_RL`, `short_LR`, `short_RL`).
