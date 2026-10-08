---
name: Radon label canonicalization
overview: Make `radon` the canonical display label/key for the Radon Transform score overlay (DF column stays `score`), cleaning up the remaining `score`-as-canonical display path in `RadonTransformPlotData.build_display_text`.
todos: []
isProject: false
---

# Canonicalize Radon score label as `radon`

## Current state

In [DecoderPredictionError.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py):

- Builder already writes the overlay string as `radon: ...` (~L1736).
- `build_display_text` (~L1570) still treats **`score` as the canonical key** (default `ordered_keys = ['score', ...]`, alias remap `radon` -> `score`), while only rewriting legacy `"score: "` prefixes to `"radon: "` at display time.

DF column name stays `score` (pipeline schema); this change is display-key/label only.

## Change

In `RadonTransformPlotData.build_display_text`:

- Default ordered keys: `['radon', 'speed', 'intercept']`
- Canonicalize included keys the other way: `'score' -> 'radon'` (so `visible_overlay_label_keys=['score', ...]` and `['radon', ...]` both show the same `radon: ...` line once)
- Keep `key_to_text` entries for both `'radon'` and `'score'` pointing at the same display text (backward compatible)
- Drop the `"score: "` -> `"radon: "` rewrite if `score_text` is always built with `radon:` (keep a one-line legacy rewrite only if still useful for old in-memory overlays)

No changes needed in the Provider builder string (`f"radon: " + ...`) or in DF column loading.

## Verify

- `included_columns=['score', 'wcorr']` and `visible_overlay_label_keys=['radon', 'wcorr']` both show `radon: <value>` (not `score:`).
- Refresh overlays after change; no duplicate radon lines when both aliases appear in `visible_overlay_label_keys`.