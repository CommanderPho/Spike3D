---
name: Child track lengths
overview: Set each decoder child’s `params['track_length_cm']` from `track_length_cm_dict` so `DecodedEpochSlicesPaginatedFigureController.add_data_overlays` gets a real `decoder_track_length`.
todos:
  - id: resolve-dict
    content: "In _subfn_prepare_plot_multi_decoders_stacked_epoch_slices, source track_length_cm_dict from params_kwargs (fallback: track_templates) and set each child’s track_length_cm from dict[a_name]"
    status: completed
  - id: overlay-fallback
    content: In parent add_data_overlays, set each child’s track_length_cm from its track_length_cm_dict before calling the child overlay
    status: completed
isProject: false
---

# Set child decoder track length from the dict

Child overlays in [`stacked_epoch_slices.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py) read a scalar:

```python
decoder_track_length: float = self.params.get('track_length_cm', None)
```

If that is `None`, heuristics are skipped. Callers (for example [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) around the `common_params_kwargs` dict) pass the per-decoder map as `params_kwargs['track_length_cm_dict']` (`{'long_LR': 214.0, ...}`).

[`PhoPaginatedMultiDecoderDecodedEpochsWindow._subfn_prepare_plot_multi_decoders_stacked_epoch_slices`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py) already writes `curr_params_kwargs['track_length_cm']`, but it takes the value from a fresh `track_templates.get_track_length_dict()` call and does not use the dict already in `params_kwargs`.

## Changes

In `_subfn_prepare_plot_multi_decoders_stacked_epoch_slices`:

- Resolve one shared map: `params_kwargs.get('track_length_cm_dict')`, and only if that is missing call `track_templates.get_track_length_dict()`.
- Keep that map on the shared `params_kwargs` so every child still receives `track_length_cm_dict`.
- For each decoder name `a_name`, set `curr_params_kwargs['track_length_cm'] = track_length_cm_dict[a_name]` (the existing assignment, sourced from the dict).

In the parent `add_data_overlays` loop (same class), before calling each child’s `add_data_overlays`, if that child has `track_length_cm_dict` and `a_name` is a key, set `a_pagination_controller.params.track_length_cm` from `track_length_cm_dict[a_name]`. That covers children that received the dict later (for example via `update_params`) without a per-decoder scalar.

The yellow-blue marginal controller is not one of the four decoder children and will be left unchanged.
