---
name: Fig2 top margin 0.78
overview: Apply the validated post-hoc `subplots_adjust(top=0.78)` to both Fig2 outputs inside `PaperFigureTwo.display`, using the existing `top_margin` parameter with default 0.78.
todos:
  - id: fig2-top-margin
    content: "In PaperFigureTwo.display: default top_margin=0.78; subplots_adjust both figs before save"
    status: completed
isProject: false
---

# Fig2 top margin clearance

## Where

[`PaperFigureTwo.display`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) (~line 1300).

## Change

1. Change default `top_margin=0.8` → `top_margin=0.78` (matches the value you validated).
2. After stats overlays / AnyC inset and **before** `save_figure`, apply:

```python
for _fig_out in [_fig_2_theta_out, _fig_2_replay_out]:
    _fig_out.fig.subplots_adjust(top=top_margin)
```

Callers can still override with `display(..., top_margin=...)`. No notebook edits.