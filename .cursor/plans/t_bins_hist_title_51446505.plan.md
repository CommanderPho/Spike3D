---
name: t_bins hist title
overview: When `data_grain` is `per_time_bin`, set the pre/post histogram x-axis titles to `# t_bins` instead of `# Events` in `plotly_pre_post_delta_scatter`.
todos:
  - id: hist-count-title
    content: Derive hist_count_axis_title from data_grain and use it for cols 1 and 3
    status: completed
isProject: false
---

# Use `# t_bins` for time-bin histogram axes

## Context

In [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) `plotly_pre_post_delta_scatter`, histogram count axes are hardcoded:

```1092:1094:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py
    fig.update_xaxes(title_text="# Events", row=1, col=1)
    fig.update_xaxes(title_text="Delta-aligned Event Time (seconds)", row=1, col=2)
    fig.update_xaxes(title_text="# Events", row=1, col=3)
```

Callers already pass `data_context` with `data_grain='per_time_bin'` or `'per_epoch'` (e.g. in [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py)). That grain is already folded into figure metadata at line 1070.

## Change

Just before the three `update_xaxes` calls, derive the hist count title from `data_context`:

```python
data_grain = data_context.to_dict().get('data_grain', None)
hist_count_axis_title: str = "# t_bins" if (data_grain == 'per_time_bin') else "# Events"
fig.update_xaxes(title_text=hist_count_axis_title, row=1, col=1)
fig.update_xaxes(title_text="Delta-aligned Event Time (seconds)", row=1, col=2)
fig.update_xaxes(title_text=hist_count_axis_title, row=1, col=3)
```

- Default remains `# Events` when `data_grain` is missing or `per_epoch`.
- Leave the middle scatter title unchanged (only `# Events` was requested).
- No caller changes.

## Scope

- Single file: [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py)
- ~4 lines around 1092–1094