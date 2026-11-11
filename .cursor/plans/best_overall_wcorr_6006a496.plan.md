---
name: best overall wcorr
overview: Add a raw `best_overall_wcorr` column (abs-max across the four decoder wcorr scores) via `MostLikelyDecoderDataframeAccessor`, merge it onto HDF5-recovered ripple `filter_epochs`, and include it in the default overlay allow-list/display path.
todos:
  - id: mld-best-overall
    content: Add take_best_overall_score_across_decoders + add_best_overall_columns on MostLikelyDecoderDataframeAccessor
    status: completed
  - id: merge-enrich
    content: Compute on scores_df in enrich; transfer best_overall_wcorr in merge_score_and_shuffle_onto_decoder_result
    status: completed
  - id: overlay-allowlist
    content: Add best_overall_wcorr to DEFAULT_OVERLAY_INCLUDED_COLUMNS, WCORR_LABEL_KEYS, and overlay formatter
    status: completed
isProject: false
---

# Add `best_overall_wcorr` to MLD accessor and default overlay

## Semantics (chosen)

Mirror shuffle’s [`best_overall_wcorr_z`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py) (`nanmax(abs(...))`), but on **raw** scores:

`best_overall_wcorr = nanmax(|wcorr_long_LR|, |wcorr_long_RL|, |wcorr_short_LR|, |wcorr_short_RL|)`

This is **not** signed `HIGHEST_wcorr` and **not** `mld_wcorr` (posterior-argmax pick). It only needs the four `{score}_{decoder}` columns (same inputs as `take_highest_score_across_decoders`).

## 1. Extend `MostLikelyDecoderDataframeAccessor`

In [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) (~6496–6549):

- Add `take_best_overall_score_across_decoders(cls, df, score_col_name, decoder_names=None)` → `np.nanmax(np.abs(score_mat), axis=1)`.
- Add `add_best_overall_columns(cls, df, score_col_names=('wcorr',), overwrite_existing=False)` → writes `best_overall_{score}` when missing (or always if overwrite).
- Add thin instance wrapper `adding_best_overall_columns(...)`.
- Document `best_overall_{score}` in the class docstring table next to `HIGHEST_{score}`.

Keep signatures single-line; two blank lines between methods.

## 2. Compute + merge onto decoder `filter_epochs`

In `HDF5RecoveredPosteriorRippleViewer`:

- In [`enrich_ripple_decoder_results_with_score_shuffle_dfs`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) (~434–439): before the per-decoder loop, ensure `scores_df` has `best_overall_wcorr` via `MostLikelyDecoderDataframeAccessor.add_best_overall_columns(scores_df, score_col_names=['wcorr'])` (only when the four `wcorr_*` cols exist; skip quietly if not).
- In [`merge_score_and_shuffle_onto_decoder_result`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) (~361–372): when building the scores right-hand merge, also transfer unsuffixed `best_overall_wcorr` from `scores_df` onto each decoder’s `filter_epochs` (same start-join as the per-decoder unsuffix rename). It is cross-decoder, so it must not go through `{base}_{decoder}` → `base` renaming.

## 3. Default overlay + display allow-list

- Append `'best_overall_wcorr'` to `DEFAULT_OVERLAY_INCLUDED_COLUMNS` (near `'wcorr'`) at line ~119.
- In [`DecoderPredictionError.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\DisplayFunctions\DecoderPredictionError.py) `OverlayLabelsPlotData`:
  - Add `'best_overall_wcorr'` to `WCORR_LABEL_KEYS` (~1967).
  - Add a formatter in `init_batch_from_epochs_df` (~2032) e.g. `best_overall_wcorr: …` so the column is not dropped for missing formatters.

```mermaid
flowchart LR
  scores_df["scores_df wcorr_long_LR ..."]
  mld["pho_mld.add_best_overall_columns"]
  best["best_overall_wcorr"]
  merge["merge onto each filter_epochs"]
  overlay["DEFAULT_OVERLAY_INCLUDED_COLUMNS"]
  scores_df --> mld --> best --> merge --> overlay
```

## Out of scope

- No change to shuffle `best_overall_wcorr_z` computation or default `sort_by`.
- No notebook edits unless you ask.
- Do not rename/replace existing `HIGHEST_*` columns.