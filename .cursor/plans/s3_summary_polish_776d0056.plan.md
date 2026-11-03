---
name: S3 summary polish
overview: "Polish Figure S3 multi-tbin summary bars only: hide legends, halve scatter marker size, and pull significance asterisks closer to their brackets—without changing Figure 4 / S8 / S9."
todos: []
isProject: false
---

# Figure S3 summary polish

Scoped to [`plot_fig2_style_summary_multi_tbin_bars`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) only.

## 1. Hide legends

Remove (or skip) both legend blocks (~3015–3040):

- Time-bin-size `Patch` legend
- Animal marker `Line2D` legend

No `ax.legend(...)` / `ax.add_artist(...)` for this plot.

## 2. Halve scatter point size

Current `scatter_kwargs` (~2952):

```python
scatter_kwargs=dict(edgecolors='white', linewidths=0.4, zorder=4)
```

Matplotlib default scatter area is `s=36` (markersize 6²). Half linear size → `s=9`:

```python
scatter_kwargs=dict(edgecolors='white', linewidths=0.4, zorder=4, s=9)
```

Passed into `PaperFigureTwo.create_plot` via existing `scatter_extra_kwargs` path.

## 3. Pull significance asterisks closer to brackets

Do **not** change global [`add_significance_bars`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) defaults (used by Fig4). For S3 calls only (~3009–3012), pass tighter offsets:

```python
add_significance_bars(..., y=bracket_y, color=..., y_offset_frac=0.04, text_additional_y_offset=0.002)
```

(`text_additional_y_offset` default is `0.02 * |y|`, which is what floats the `*` too far above the bracket top.)

## Out of scope

- Figure 4 / S8 / S9 four-bar legends or scatter sizes
- Global `add_significance_bars` API changes