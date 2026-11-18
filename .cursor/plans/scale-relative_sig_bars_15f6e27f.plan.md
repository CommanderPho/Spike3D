---
name: Scale-relative sig bars
overview: Make significance-bar vertical spacing a fraction of the data max so participation plots lose the large top gap while Hz-mode plots keep proportional headroom.
todos:
  - id: frac-y-offset
    content: In add_significance_bars, default y_offset/text offsets from fractions of |y| (0.08 / 0.02); keep absolute overrides
    status: completed
isProject: false
---

# Scale-relative significance-bar offsets

## Cause

[`add_significance_bars`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) defaults to absolute `y_offset=0.5`. That is ~50% of a 0–1 participation axis and ~1% of a tens-of-Hz axis.

## Change (single place)

Update `add_significance_bars` only in [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py):

- Add `y_offset_frac: float = 0.08`.
- Change `y_offset` default to `None`. When `None`, set `y_offset = y_offset_frac * max(abs(y), 1e-12)`.
- Change `text_additional_y_offset` default to `None`. When `None`, set `text_additional_y_offset = 0.02 * max(abs(y), 1e-12)`.
- Explicit float overrides still win for both params.

Existing Fig2 call sites (`y=max_laps_y` / `y=max_PBEs_y`) need no edits; they pick up proportional spacing on replot.

Expected spacing:
- Participation `y≈1` → bracket ~0.08 above data
- Hz `y≈50` → bracket ~4 above data (small absolute gap today becomes modest proportional headroom)