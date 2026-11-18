---
name: S3 grouped hist
overview: For Figure S3 only (`color_by_time_bin_size`), switch side histograms to opaque side-by-side grouped bars while keeping density normalization; leave Figure 4 / S8 / S9 on translucent overlay + raw counts.
todos: []
isProject: false
---

# Figure S3 grouped opaque histograms

## Choice

Use Plotly `barmode='group'` with `opacity=1.0` for S3 density histograms so each `time_bin_size` series is readable and comparable. Do **not** use stack (misleading with `histnorm`).

## Edits

### 1. S3 hist kwargs — [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py)

In `_perform_plot_iteratively_constrained` (~3306):

```python
_tbin_hist_kwargs = dict(color='time_bin_size', histnorm='probability density', barmode='group', opacity=1.0)
```

In `_helper_perform_plot_pre_post_delta_scatter` when `color_by_time_bin_size` (~7882):

```python
hist_kwargs['color'] = "time_bin_size"
hist_kwargs['histnorm'] = 'probability density'
hist_kwargs['barmode'] = 'group'
hist_kwargs['opacity'] = 1.0
```

Non-S3 branches unchanged (no `barmode`/`opacity` override → defaults stay overlay + 0.5).

### 2. Honor hist `barmode` in layout — [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py)

Replace the hard-coded layout override (~1050):

```python
fig.update_layout(yaxis=dict(range=forced_range_y), barmode='overlay')
```

with:

```python
_active_barmode = hist_kwargs.get('barmode', 'overlay')
fig.update_layout(yaxis=dict(range=forced_range_y), barmode=_active_barmode)
```

Default remains `'overlay'` when callers omit `barmode`, so Figure 4 / S8 / S9 are unchanged.

Shared pre/post density axis logic stays as-is (uses plotted bar heights when `histnorm` is set).

## Scope

- No changes to scatter coloring, legend, stats, or other figure configs.
- No matplotlib / ridge / facet alternatives.