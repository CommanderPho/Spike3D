---
name: Export Pseudo2D H5
overview: Enrich batch decoded-posterior HDF5 exports so they also write the true merged Pseudo2D `DecodedFilterEpochsResult` under decoder key `pseudo2D`, alongside the existing four independent 1D decoders — by extending the existing saver, with no new helper function.
todos:
  - id: extend-perform-save
    content: Extend perform_save_all_decoded_posteriors_to_HDF5 to accept optional Pseudo2D lap/ripple results and inject decoder key 'pseudo2D'; pass save_full_object=True in to_hdf
    status: completed
  - id: wire-batch-sites
    content: Pass DirectionalMergedDecoders all_directional_* results into both batch H5 export call sites
    status: completed
  - id: sanity-check
    content: Confirm enriched dict keys include pseudo2D and to_hdf receives save_full_object=True
    status: completed
isProject: false
---

# Export true Pseudo2D in batch posterior H5

## Context

Batch currently saves only the four independent 1D results from `DirectionalDecodersEpochsEvaluations` via [`PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\data_exporting.py) (call sites ~1006 and ~2636 in [`batch_user_completion_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py)).

The true joint Pseudo2D already lives on `DirectionalMergedDecoders`:

- `all_directional_laps_filter_epochs_decoder_result`
- `all_directional_ripple_filter_epochs_decoder_result`

each epoch’s `.p_x_given_n` is `(n_x, 4, n_t)`.

`perform_save_all_decoded_posteriors_to_HDF5` already loops over whatever keys are in the dicts. No separate inject helper is needed.

`SingleEpochDecodedResult.to_hdf(..., save_full_object=True)` writes the full array to `p_x_given_n_full` and the x-marginal to legacy `p_x_given_n`. `read_hdf` prefers `p_x_given_n_full`, so recover paths pick up true Pseudo2D under decoder key `pseudo2D`.

## Approach (no new helper)

1. Extend existing [`perform_save_all_decoded_posteriors_to_HDF5`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\data_exporting.py) with two optional kwargs:

```python
pseudo2D_laps_filter_epochs_decoder_result: Optional[DecodedFilterEpochsResult] = None,
pseudo2D_ripple_filter_epochs_decoder_result: Optional[DecodedFilterEpochsResult] = None,
```

At the start of the function (after deepcopying or before the per-epochs save loops), if a Pseudo2D result is provided, insert it into the corresponding dict under key `'pseudo2D'` (skip / warn if that key already exists). Existing callers that omit the kwargs behave unchanged.

2. In `save_decoded_posteriors_to_HDF5`, pass `save_full_object=True` explicitly into `to_hdf(...)` so Pseudo2D stores `p_x_given_n_full` (3D).

3. At both batch H5 export sites, pass the MergedDecoders results into the existing function:

```python
directional_merged_decoders_result = curr_active_pipeline.global_computation_results.computed_data.get('DirectionalMergedDecoders')
...
PosteriorExporting.perform_save_all_decoded_posteriors_to_HDF5(
    ...,
    pseudo2D_laps_filter_epochs_decoder_result=getattr(directional_merged_decoders_result, 'all_directional_laps_filter_epochs_decoder_result', None),
    pseudo2D_ripple_filter_epochs_decoder_result=getattr(directional_merged_decoders_result, 'all_directional_ripple_filter_epochs_decoder_result', None),
)
```

(Use `.get('DirectionalMergedDecoders')` so a missing result only skips Pseudo2D, not the whole export.)

## HDF layout after change

Under existing `.../save_decoded_posteriors_to_HDF5/`:

- `long_LR|long_RL|short_LR|short_RL` / `{laps|ripple}` / `{epoch}` (unchanged 1D)
- **`pseudo2D` / `{laps|ripple}` / `{epoch}`** with `p_x_given_n_full` shape `(n_x, 4, n_t)`

## Usage after re-export

```python
restored, _ = HDF5RecoveredPosteriorRippleViewer.recover_single_epoch_decoded_results_from_hdf5(h5_path)
p = restored['pseudo2D']['ripple']['0'].p_x_given_n  # (n_x, 4, n_t)
plot_map_based_context_decoder_walkthru(p, save_path=..., context_names=['long_LR','long_RL','short_LR','short_RL'])
```

## Out of scope

- New helper function (`include_directional_pseudo2D_decoder_results` — dropped)
- Changing filename / tbin naming (still driven by `DirectionalDecodersEpochsEvaluations` ripple tbin)
- Re-exporting old H5s (new runs only)
- Bapun `contextual_pf2D` export sites (~3830) — already a different decoder key path
