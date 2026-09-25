---
name: scatter x bounds
overview: Pin the scatter subplot x-axis to the provided time_delta bounds inside plotly_pre_post_delta_scatter, and only pass those bounds from PlotlyHelpers when both are set.
todos:
  - id: pin-scatter-x
    content: In plotly_pre_post_delta_scatter time_delta_tuple block, update_xaxes range on row=1 col=2 to delta_relative bounds
    status: completed
  - id: helpers-pass-bounds
    content: "PlotlyHelpers: only pass time_delta_tuple when both earliest/latest are set; property returns None otherwise"
    status: completed
isProject: false
---

# Pin scatter x-range to delta time bounds

## Cause

White flanks are empty scatter (col=2) x-margins: epoch `vrect`s use `time_delta_tuple`, but col=2 is left to autorange/stale range. Hist count axes are unrelated.

## Change 1 — apply range in [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py)

In `plotly_pre_post_delta_scatter`, inside the existing `if time_delta_tuple is not None:` block (~1074–1083), after computing `delta_relative_t_start` / `delta_relative_t_end` and adding epoch shapes, pin the scatter x-axis to those same bounds:

```python
fig.update_xaxes(range=[delta_relative_t_start, delta_relative_t_end], autorange=False, row=1, col=2)
```

Use the delta-relative values (same as the vrects). When callers pass `t_delta=0.0`, this equals `[earliest_delta_aligned_t_start, latest_delta_aligned_t_end]`.

Leave hist x updates and the later col=2 title `update_xaxes` alone (title call does not clear `range`).

## Change 2 — pass bounds from [`plotly_templates.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\plotly_templates.py) `PlotlyHelpers` only when present

`_perform_plot_pre_post_delta_scatter` currently always builds a tuple, even when `self.earliest_delta_aligned_t_start` / `self.latest_delta_aligned_t_end` are `None`.

Tighten it to:

1. Resolve `time_delta_tuple` from kwargs if provided.
2. Else, if both `earliest_delta_aligned_t_start` and `latest_delta_aligned_t_end` are non-`None` (kwargs override, else `self.*`), build `(earliest, 0.0, latest)`.
3. Else pass `time_delta_tuple=None` so shapes/range are skipped.

Also make `time_delta_tuple` property return `Optional[Tuple[...]]` — `None` when either bound is missing — so it matches that gate.

## Out of scope

- No caller changes in `PhoDiba2023Paper.py` (they already pass a real `time_delta_tuple`).
- Leave the DataFrameFilter `update_xaxes(col=2, range=...)` line as-is (redundant once Change 1 lands, still harmless on refresh).
