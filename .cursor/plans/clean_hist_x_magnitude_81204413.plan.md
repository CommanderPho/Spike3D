---
name: Clean hist x magnitude
overview: Restore the previous vertical-axis / hist-bin code unchanged, and implement shared pre/post histogram magnitude as a small x-axis-only change (count range on cols 1 and 3).
todos:
  - id: restore-hist-loops
    content: Restore pre/post hist loop bodies (xbins + TODO); remove shared-ybins helpers
    status: completed
  - id: x-only-magnitude
    content: Track hist_count_max from trace.y via np.histogram; apply update_xaxes on cols 1 and 3 only
    status: completed
isProject: false
---

# Clean shared histogram count (x-axis only)

## Goal

In [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) `plotly_pre_post_delta_scatter`: make pre/post histogram **counts** comparable. Do **not** touch figure Y limits or hist bin edge rewriting for the probability axis.

Vertical axis stays as it already is later in the function:

```python
fig.update_yaxes(row=1, range=[-0.05, 1.05], autorange=False, fixedrange=True)
```

## Approach

Restore the pre/post hist loop bodies to the prior shape (including the existing `a_trace.xbins...` assignments and the post-delta TODO comment). Remove the new `_subfn_apply_shared_hist_ybins` / shared-`ybins` block that was altering bin edges and interacting with the locked Y range.

Replace the incomplete `hist_y_min` / `hist_y_max` stub with a minimal count-magnitude path:

1. When `should_set_hist_same_magnitude_axes`, init `hist_count_max = None`.
2. After each hist trace is configured/added (pre col=1, post col=3), update `hist_count_max` from that trace’s raw `y` values via `np.histogram` using only `hist_kwargs`’s `nbins` / `range_y` (fallback: `histogram_bins`, `[0.0, 1.0]`). This estimates bar lengths for overlay mode; it does not call `update_yaxes` or rewrite `ybins`.
3. After both hists are added, if `hist_count_max > 0`, apply only:

```python
fig.update_xaxes(range=[0, hist_count_max], autorange=False, row=1, col=1)
fig.update_xaxes(range=[0, hist_count_max], autorange=False, row=1, col=3)
```

Leave scatter x (col=2) and all existing Y-axis updates alone.

## Scope

- Single file: [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py)
- No changes to callers in `PhoDiba2023Paper.py`
- No changes to the later `update_yaxes(... range=[-0.05, 1.05] ...)` block
