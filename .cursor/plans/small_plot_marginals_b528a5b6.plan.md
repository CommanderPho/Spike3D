---
name: Small plot marginals
overview: Add top/right black marginal histograms to `_plot_single_track_firing_rate_compare` (the two small Fig3 panels), matching the big FRI plot’s marginal styling, enabled by default when `prepare_for_publication=True`.
todos:
  - id: enable-hist-flag
    content: Add enable_histograms resolve (default on for prepare_for_publication)
    status: completed
  - id: mosaic-layout
    content: Use subplot_mosaic with ax_histx/ax_scatter/ax_histy when histograms enabled
    status: completed
  - id: draw-style-hists
    content: Draw black top/right hists with FRI-matching spine/tick cleanup and Hz-appropriate bins
    status: completed
isProject: false
---

# Marginal histograms on small Fig3 panels

## Scope
Only [`_plot_single_track_firing_rate_compare`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\DisplayFunctions\MultiContextComparingDisplayFunctions\LongShortTrackComparingDisplayFunctions.py) (~3071). Top + right black marginals (no diagonal histogram — that is FRI-index-specific). Notebook / `AcrossSessionsVisualizations` need no changes: kwargs already forward, and pub mode will turn histograms on automatically.

## Behavior
- Add `enable_histograms=None` (or pop from `scatter_params`).
- Resolve: if `None`, use `True` when `prepare_for_publication` else `False` (non-pub layout unchanged).
- Explicit `enable_histograms=True/False` always wins.

## Layout (match big plot mosaic style)
When histograms enabled, replace `plt.subplots(...)` with the same mosaic pattern as `_plot_long_short_firing_rate_indicies` (~2238), without the diagonal cell:

```python
fig = plt.figure(**fig_kwargs)
ax_dict = fig.subplot_mosaic(
    [["ax_histx", "."],
     ["ax_scatter", "ax_histy"]],
    height_ratios=[1, 8],
    width_ratios=[8, 1],
    gridspec_kw=dict(wspace=0, hspace=0),
)
ax = ax_dict["ax_scatter"]
```

When histograms disabled, keep current `fig, ax = plt.subplots(**fig_kwargs)`.

## Histogram drawing (mirror big plot ~2476–2511)
After scatter + axes styling, if `enable_histograms`:
- `ax_histx.sharex(ax)`, `ax_histy.sharey(ax)`
- Hide shared tick labels on hist axes
- Build bins from Laps/PBE Hz values (non-negative):  
  `binwidth = max(xymax / 20.0, 1e-6)` then `bins = np.arange(0, lim + binwidth, binwidth)` (not the FRI ±0.075 index bins)
- `ax_histx.hist(x, bins=bins, color='black')`
- `ax_histy.hist(y, bins=bins, orientation='horizontal', color='black')`
- Same cleanup as big plot: no spines, no tick marks/labels on hist axes

## Preserve the rest
Titles, Artboard tick/spine styling, purple R/fit, return `(fig, ax, active_display_context)` unchanged (`ax` remains the scatter axes).
