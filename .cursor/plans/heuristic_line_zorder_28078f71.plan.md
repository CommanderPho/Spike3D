---
name: Heuristic line zorder
overview: Put wcorr/radon overlay labels above heuristic sequence lines by raising label zorder and lowering the intentionally high heuristic hline zorder in the stacked-epoch overlay path.
todos:
  - id: raise-label-zorder
    content: Set OVERLAY_LABEL_ZORDER=50 on wcorr and radon AnchoredText after create/update in DecoderPredictionError.py
    status: completed
  - id: lower-hline-zorder
    content: Change heuristic sequence_position_hlines zorder from 10 to 5 in DecodedSequenceAndHeuristicsPlotDataProvider
    status: completed
isProject: false
---

# Heuristic lines below overlay labels

## Cause

In the stacked epoch view, heuristic sequence overlays are drawn by [`DecodedSequenceAndHeuristicsPlotDataProvider`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py) with:

```python
sequence_position_hlines_kwargs=dict(..., zorder=10, ...)  # intentionally "on-top"
```

Wcorr/radon labels are created via `add_inner_title(...)` with **no** `zorder`, so they stay at matplotlib’s default (~0). Result: yellow heuristic hlines cross through the top-right `wcorr:` / `radon:` boxes.

Related: direction-change segments in [`heuristic_replay_scoring.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/heuristic_replay_scoring.py) default to `zorder=22`, so raising labels above that is required for a complete fix.

## Approach

Single file change in [`DecoderPredictionError.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py):

1. **Raise overlay label zorder** after each successful create/update of the anchored text in both:
   - Radon `_callback_update_curr_single_epoch_slice_plot` (~L1960–2003)
   - WCorr `_callback_update_curr_single_epoch_slice_plot` (~L2510–2552)

   Immediately before storing into `plots[...]`, if the artist is non-`None`:

   ```python
   anchored_text.set_zorder(50)
   ```

   Use a shared module-level constant (e.g. `OVERLAY_LABEL_ZORDER = 50`) so both providers stay aligned and sit above heuristic hlines (10/5) and direction-change lines (22).

2. **Lower heuristic sequence hline zorder** in `DecodedSequenceAndHeuristicsPlotDataProvider._callback_update_curr_single_epoch_slice_plot` (~L3260):

   - Change `zorder=10` → `zorder=5` in `sequence_position_hlines_kwargs`
   - Keep dots at `zorder=11` (markers stay above lines; labels at 50 stay above both)

No changes to `add_inner_title` / NeuroPy unless we later want a general `zorder=` kwarg; calling `set_zorder` at the call sites is the minimal fix.

## Verify

Reload the stacked epoch page (or `refresh_current_page()`). Yellow heuristic segments that pass through the top-right corner should sit behind the `wcorr`/`radon` boxes; labels remain fully readable.