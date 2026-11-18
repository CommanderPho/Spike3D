---
name: Non-sci x-axis seconds
overview: Force paginated DecodedEpochSlices x-axes (absolute time in seconds) to use sparse non-scientific, non-offset ticks so screen and single-axes exports never show packed or `1.117e3`-style labels.
todos: []
isProject: false
---

# Non-scientific sparse seconds on DecodedEpochSlices x-axis

## Problem
Absolute epoch times (~10²–10⁴ s) make Matplotlib’s default formatter switch to scientific/offset notation on the time axis, and short epochs still get too many major ticks so labels collide. Data are already in seconds; only tick locator/formatting is wrong. Exports that crop a single `ax` inherit that formatting.

## Change
In [`stacked_epoch_slices.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py), inside `DecodedEpochSlicesPaginatedFigureController.on_jump_to_page` immediately after `curr_ax.set_xlim(*curr_epoch_slice)` (~L1499), apply the same pattern already used by the Silx debugger (`RadonTransformDebugger._configure_plot_display`):

Extract a tiny private helper on the controller:

```python
@classmethod
def _configure_time_axis_seconds_formatter(cls, curr_ax, nbins: int = 6):
    """ Absolute time in seconds; sparse ticks; never scientific / offset notation. """
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    curr_ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    curr_ax.xaxis.set_major_formatter(fmt)
```

Call it once per visible subplot after `set_xlim`. Default `nbins=6` matches the Silx debugger; override via `params.setdefault('time_axis_max_nbins', 6)` if present so callers can tune density without code changes.

No change to data units or export notebook code — re-`jump_to_page` / `draw` before `savefig` picks up the locator/formatter.

## Out of scope
- Changing Silx debugger (already correct)
- Forcing a specific `xlabel` string unless already present
- Altering y-axis formatting

## Verify
On an epoch with `start_t ≈ 1117`, subplot x ticks show a handful of plain decimals (e.g. `1117.4`) not `1.117e+03` / offset `+1.117e3`, and labels do not overlap; single-ax PDF/SVG export matches.
