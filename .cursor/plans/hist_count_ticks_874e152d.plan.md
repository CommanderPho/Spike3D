---
name: hist count ticks
overview: Pin both pre/post histogram count (x) axes to exactly three matching tick values and a shared slant angle inside the existing shared-magnitude block of `plotly_pre_post_delta_scatter`.
todos:
  - id: hist-count-ticks
    content: Add tickmode/tickvals/tickangle to shared hist x-axis updates for cols 1 and 3
    status: completed
isProject: false
---

# Sync hist count axes to 3 slanted ticks

## Context

In [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) `plotly_pre_post_delta_scatter`, shared count ranges are already applied when `should_set_hist_same_magnitude_axes` is True:

```1043:1050:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py
    if should_set_hist_same_magnitude_axes:
        # Shared # Events (x) range for both horizontal histograms — leave scatter time axis (col=2) alone
        if (hist_count_max is not None) and (hist_count_max > 0):
            shared_hist_x_range = [0, hist_count_max]
            ...
            fig.update_xaxes(range=shared_hist_x_range, autorange=False, row=1, col=1)
            fig.update_xaxes(range=shared_hist_x_range, autorange=False, row=1, col=3)
```

Plotly still auto-chooses tick count (3 vs 4) and angle (0° vs 90°) per subplot, which matches the screenshot mismatch.

## Change

Extend those two `update_xaxes` calls to also set explicit ticks and slant:

```python
hist_count_tickvals = [0, hist_count_max // 2, hist_count_max]
fig.update_xaxes(
    range=shared_hist_x_range, autorange=False,
    tickmode='array', tickvals=hist_count_tickvals,
    tickangle=-45,
    row=1, col=1,
)
fig.update_xaxes(
    range=shared_hist_x_range, autorange=False,
    tickmode='array', tickvals=hist_count_tickvals,
    tickangle=-45,
    row=1, col=3,
)
```

- `tickmode='array'` + `tickvals` guarantees exactly three identical labels on both hists (unlike `nticks`, which is only a hint).
- `tickangle=-45` slant both count axes the same way.
- Integer mid tick via `// 2` keeps labels clean (e.g. `[0, 500, 1000]`).
- Leave scatter x (col=2) and all y-axis updates untouched.
- No caller changes; gated by existing `should_set_hist_same_magnitude_axes`.

## Scope

- Single file: [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py)
- ~6 lines in the shared-magnitude block (~1043–1050)
