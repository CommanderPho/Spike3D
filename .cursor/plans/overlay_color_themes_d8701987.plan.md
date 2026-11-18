---
name: Overlay color themes
overview: Theme wcorr labels blue and heuristic sequence graphics green (matching radon’s yellow stroke style), and tighten radon stack padding so the gap under mseq_tcov matches inter-line spacing in the wcorr block.
todos:
  - id: wcorr-blue-stroke
    content: Set wcorr AnchoredCustomText stroke_foreground to bright blue theme color
    status: completed
  - id: heuristic-green-cmap
    content: Add subsequence_cmap kwarg; pass greens cmap from DecodedSequenceAndHeuristicsPlotDataProvider
    status: completed
  - id: radon-pad-gap
    content: Reduce pad_axes default from 0.02 to 0.0 for radon stacking under wcorr
    status: completed
isProject: false
---

# Overlay color themes + radon stack gap

## Goals

1. **Wcorr label** outline → bright blue (same stroke style as radon’s yellow).
2. **Heuristic plot elements** (triangle fills, subsequence number outlines, and matching sequence hlines) → green theme.
3. **Fix unequal gap** between the last wcorr/heuristic text line (`mseq_tcov`) and `radon:` by reducing stack pad.

## Color constants

In [`DecoderPredictionError.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py):

- Keep radon: `#ffee00`
- Add wcorr theme: `#00B0FF` (bright blue stroke, black text — mirrors radon)
- Add heuristic theme: `#00C853` (main) with a short green family for secondary subsequences

## 1. Wcorr label → blue outline

In `WeightedCorrelationPaginatedPlotDataProvider._callback_update_curr_single_epoch_slice_plot` (~L2526), replace:

```python
stroke_foreground='grey'
```

with the wcorr theme blue (`#00B0FF`), keep `strokewidth=1.5`, `stroke_alpha=0.75`, `text_foreground='black'` — same pattern as radon’s `_subfn_build_kwargs`. Store the color as a class attr (e.g. `theme_stroke_color`) next to the existing unused `text_color` for consistency.

Heuristic score *text* columns (`coverage`, `mseq_*`, …) stay in this same AnchoredCustomText block (blue outline). Plot-side heuristic graphics get the green theme below.

## 2. Heuristic triangles / numbers / hlines → green

Today [`plot_time_bins_multiple`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py) hardcodes `cmap = plt.get_cmap('tab10')` (~L1822), so the main sequence is tab10 blue; number outlines pull from that color via `subseq_idx_text_outline_color=('color','color','color',0.95)`.

- Add kwarg `subsequence_cmap=None` (default keeps `'tab10'` for other callers).
- When provided, use that cmap instead of tab10.
- From `DecodedSequenceAndHeuristicsPlotDataProvider._callback_update_curr_single_epoch_slice_plot` (~L3300), pass a greens cmap (e.g. custom `ListedColormap` of a few greens with longest/main = `#00C853`) via `common_plot_time_bins_multiple_kwargs`.

Triangle `scatter(..., color=color)` and number strokes then follow green automatically; sequence hlines already use the same `color`.

## 3. Fix radon vs wcorr gap

Unequal gap is from `pad_axes=0.02` in `_axes_y_below_anchored_artist` / `_reposition_radon_text_below_wcorr` / `_resolve_radon_label_bbox_y` — ~2% of axes height on short epoch panels, larger than in-block line spacing.

- Change default `pad_axes` to `0.0` (window extent already includes AnchoredText `borderpad`; extra pad is what creates the visual hole).
- Keep `radon_label_bbox_y` as measurement-failure fallback only.

## Files

- [`DecoderPredictionError.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py) — wcorr stroke; heuristic greens cmap pass-through; pad default
- [`heuristic_replay_scoring.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py) — `subsequence_cmap` kwarg in `plot_time_bins_multiple`

## Verify

```python
paginated_multi_decoder_decoded_epochs_window.add_data_overlays(
    included_columns=['radon', 'wcorr', 'coverage', 'mseq_tcov'])
```

Expect: wcorr block with blue outline; green triangles + green number strokes; radon yellow outline stacked with spacing similar to wcorr line gaps (no large hole under `mseq_tcov`).