---
name: hist scatter label gap
overview: Fix the left-hist max count tick colliding with the scatter’s leftmost time tick by reversing the left hist x-axis so large labels sit on the outer edge, and slightly widen subplot horizontal spacing.
todos:
  - id: reverse-left-hist
    content: Set left hist x range to [nice_hist_count_max, 0]; keep right as [0, max]
    status: completed
  - id: widen-h-spacing
    content: Bump horizontal_spacing 0.01 → 0.025 in _helper_build_pre_post_delta_figure_if_needed
    status: completed
isProject: false
---

# Fix hist / scatter x-label collision

## Cause

In [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) `plotly_pre_post_delta_scatter`, both hist count axes use `range=[0, nice_hist_count_max]`. On the **left** hist that puts the max tick next to the scatter’s min time tick (`-2000`). Subplot `horizontal_spacing=0.01` leaves almost no gutter for those two labels.

## Approach (chosen default)

**Reverse the left hist count axis** so ticks read max → 0 toward the scatter (bars grow outward). Keep the right hist as `0 → max`. Slightly increase `horizontal_spacing` so `0` vs `-2000` still has a little air.

This is the standard mirrored-marginal pattern and keeps all three tick labels visible.

## Change 1 — reverse left hist range

In the `should_set_hist_same_magnitude_axes` block (~1073–1074):

```python
# Left: max on outside, 0 next to scatter (avoids max vs -2000 collision)
fig.update_xaxes(range=[nice_hist_count_max, 0], autorange=False, tickmode='array', tickvals=hist_count_tickvals, tickangle=tickangle, row=1, col=1)
# Right: unchanged orientation
fig.update_xaxes(range=shared_hist_x_range, autorange=False, tickmode='array', tickvals=hist_count_tickvals, tickangle=tickangle, row=1, col=3)
```

`tickvals` stay `[0, mid, max]`; Plotly places them on the reversed range.

## Change 2 — widen subplot gutter

In `_helper_build_pre_post_delta_figure_if_needed` (~407–409), change `horizontal_spacing=0.01` → `0.025` on both `make_subplots` / `set_subplots` calls.

Note: extant `FigureWidget`s keep their old domains until recreated; new figures and full rebuilds pick up the wider gap. The axis reverse (Change 1) applies on every plot pass including refresh.

## Out of scope

- No changes to scatter `range` pinning or nice-tick math
- No caller / template changes
- Leave `tickangle` as-is (currently `90`)
