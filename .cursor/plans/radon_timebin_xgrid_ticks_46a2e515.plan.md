---
name: Radon timebin xgrid ticks
overview: Add DecodedEpochSlices-style vertical time-bin edge lines to RadonTransformDebugger, and fix colliding x-tick labels by limiting major ticks with MaxNLocator while keeping non-scientific formatting.
todos:
  - id: add-xgrid-method
    content: Add should_draw_time_bin_boundaries fields + add_time_bin_xgrid (silx Curves); call from refresh_overlays
    status: completed
  - id: fix-xtick-density
    content: Use MaxNLocator(nbins=6) + non-scientific ScalarFormatter in _configure_plot_display; re-apply before PDF save
    status: completed
isProject: false
---

# Time-bin xgrid and non-colliding x ticks in RadonTransformDebugger

## Findings

**Reference xgrid** ([`stacked_epoch_slices.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\stacked_epoch_slices.py) ~1413–1484): DecodedEpochSlices draws one `axvline` per `time_bin_container.edges` with:

```python
time_bin_edges_display_kwargs = dict(color='grey', alpha=0.5, linewidth=1.5)
```

Those lines are the bin boundaries; major tick *labels* stay sparse via matplotlib’s default locator (screenshot: labels every ~0.025–0.05 s, not every bin).

**Why Radon PDFs collide today** ([`_configure_plot_display`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py)):

```python
an_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
```

No locator / `nbins` limit. Short epochs + AutoLocator + fixed `%.2f` produce dense overlapping labels. DecodedEpochSlices does not specially set a locator; the visual separation of grid-vs-labels is what matters.

**Silx constraint:** pure matplotlib `axvline` can be wiped by `replot()` / silx redraw (used in export). Use silx `addCurve` vertical segments so the grid survives GUI refresh and PDF export.

## Changes — only [`RadonTransformDebuggerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py)

### 1. Fields (match DecodedEpochSlices params)

```python
should_draw_time_bin_boundaries: bool = field(default=True)
time_bin_edges_display_kwargs: Dict = field(default=Factory(lambda: dict(color='grey', alpha=0.5, linewidth=1.5)))
```

### 2. `add_time_bin_xgrid(self, a_plot, legend_key='time_bin_edge')`

- Read `self.result.time_bin_edges[self.active_epoch_idx]` and y-span from `self.xbin` (`[xbin[0], xbin[-1]]` or last edge).
- Remove existing items whose legend starts with `legend_key` (same pattern as scoring-band cleanup).
- If `should_draw_time_bin_boundaries`, for each edge add a silx curve: `x=[edge, edge]`, `y=[y0, y1]`, `replace=False`, `z` just above the posterior (e.g. `z=0.5`), color/alpha/linewidth from `time_bin_edges_display_kwargs`.
- Call from `refresh_overlays` after `add_real_space_posterior`, before scoring band / line.

### 3. Fix x-tick density in `_configure_plot_display`

Keep non-scientific decimals; limit how many major labels appear:

```python
from matplotlib.ticker import MaxNLocator, ScalarFormatter

an_ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
fmt = ScalarFormatter(useOffset=False)
fmt.set_scientific(False)
an_ax.xaxis.set_major_formatter(fmt)
```

`nbins=6` matches the sparse major labels in the reference screenshot; bin edges remain shown by the xgrid, not by tick marks at every bin.

Re-apply `_configure_plot_display` at the end of `_save_publication_pdf` (after `replot` / `draw`, before `saveGraph`) so export PDFs keep the locator/formatter.

## Out of scope

No notebook edits. Scoring band, trajectory, and score label unchanged.
