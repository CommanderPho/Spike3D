---
name: Overlay label field visibility
overview: Add a display-time filter param so notebook code can show/hide individual overlay label fields (e.g. only `score` and `wcorr`) via `update_params` + `refresh_current_page`, without rebuilding overlays.
todos:
  - id: radon-build-display-text
    content: Add included_keys filter to RadonTransformPlotData.build_display_text
    status: completed
  - id: wcorr-build-display-text
    content: Add included_keys filter to WeightedCorrelationPlotData.build_display_text
    status: completed
  - id: callbacks-read-param
    content: Wire visible_overlay_label_keys from params in both update callbacks; hide empty filtered labels
    status: completed
  - id: update-params-doc
    content: Document visible_overlay_label_keys example on update_params
    status: completed
isProject: false
---

# Overlay label field visibility

## Approach

Filter label text at **render time** from a new params key, keeping existing master toggles and overlay data loading unchanged.

- New param: `visible_overlay_label_keys: Optional[List[str]] = None`
  - `None` (default): show all fields already present in overlay data (current behavior)
  - e.g. `['score', 'wcorr']`: show only those lines
- Master flags stay: `enable_radon_transform_info` / `enable_weighted_correlation_info` still gate whole groups (and the radon line)
- No rebuild via `add_data_overlays` / `remove_data_overlays` required for visibility changes

Notebook usage after the window exists:

```python
paginated_multi_decoder_decoded_epochs_window.update_params(visible_overlay_label_keys=['score', 'wcorr'])
paginated_multi_decoder_decoded_epochs_window.refresh_current_page()

# restore all fields
paginated_multi_decoder_decoded_epochs_window.update_params(visible_overlay_label_keys=None)
paginated_multi_decoder_decoded_epochs_window.refresh_current_page()
```

Also passable at create time in `params_kwargs`.

## Files / changes

Primary: [`DecoderPredictionError.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py)

1. **`RadonTransformPlotData.build_display_text`** — accept optional `included_keys`; map `score`/`speed`/`intercept` to existing `*_text` fields; skip missing/empty; keep `extra_text` when present.
2. **`WeightedCorrelationPlotData.build_display_text`** — accept optional `included_keys`; filter `data_formatted_strings_dict` by key before joining.
3. **Both page-update callbacks** (`RadonTransformPlotDataProvider._callback_update_curr_single_epoch_slice_plot`, `WeightedCorrelationPaginatedPlotDataProvider` equivalent):
   - Read `visible_keys = params.get('visible_overlay_label_keys', None)`
   - Build `final_text = ...build_display_text(included_keys=visible_keys)`
   - If master enable is True but filtered `final_text` is empty: do not create/keep the AnchoredText (hide label only)
   - Radon **line** still follows only `enable_radon_transform_info` (not the key filter)

Secondary (doc only): [`stacked_epoch_slices.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py) — extend `PhoPaginatedMultiDecoderDecodedEpochsWindow.update_params` docstring with the label-keys example.

## Non-goals

- No fix to radon `included_columns` hard-unpack TODO (separate from display filtering)
- No toolbar UI / checkboxes
- No change to `add_data_overlays` column loading semantics

## Key names

| Overlay | Keys |
|---|---|
| Radon (green) | `score`, `speed`, `intercept` |
| Wcorr (right) | `wcorr`, `P_decoder`, `pearsonr`, plus any other keys already in `data_formatted_strings_dict` |
