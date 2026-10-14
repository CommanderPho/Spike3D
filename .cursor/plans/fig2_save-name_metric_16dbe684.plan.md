---
name: Fig2 save-name metric
overview: Overwrite each Fig2 subplot’s IdentifyingContext `display_fn_name` from the ylabel-derived metric so PNG/SVG saves use `inst_FR_bar_graphs` vs `Particp_bar_graphs`.
todos:
  - id: save-name-from-metric
    content: In create_plot, overwrite active_context.display_fn_name from ylabel-derived metric (inst_FR_bar_graphs vs Particp_bar_graphs)
    status: completed
isProject: false
---

# Fig2 Save Names Reflect Plot Metric

## Problem
[`PaperFigureTwo.display`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) is decorated with:

```python
@providing_context(fig='2', display_fn_name='inst_FR_bar_graphs')
```

That stamps both Laps and PBE figure contexts. Saves go through `perform_save` / `AcrossSessionResults.output_figure` using `_fig_*_out.context`, so participation plots still get filenames like `10_2_inst_FR_bar_graphs_*.png`.

## Approach
Reuse the existing ylabel → metric mapping already used for titles (`_metric_name_from_ylabel`). In `create_plot`, after `metric_name` is computed, overwrite the figure context’s `display_fn_name` before attaching it to the returned `MatplotlibRenderPlots`.

Mapping (exact strings from your examples):
- `'Firing Rates'` → `inst_FR_bar_graphs`
- `'Participation Fractions'` → `Particp_bar_graphs`

```python
_display_fn_name: str = 'Particp_bar_graphs' if (metric_name == 'Participation Fractions') else 'inst_FR_bar_graphs'
active_context = active_context.overwriting_context(display_fn_name=_display_fn_name)
```

Keep `@providing_context(..., display_fn_name='inst_FR_bar_graphs')` as the FR default; per-subplot overwrite handles mixed Laps/PBE metrics.

## File
[`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) — only `create_plot` (optionally a one-line helper next to `_metric_name_from_ylabel` for clarity).

No changes to `AcrossSessionResults.output_figure` or save plumbing.

## Result
- FR ylabel → `..._2_inst_FR_bar_graphs_Laps_...` / `..._PBE_...`
- Participation ylabel → `..._2_Particp_bar_graphs_Laps_...` / `..._PBE_...`
