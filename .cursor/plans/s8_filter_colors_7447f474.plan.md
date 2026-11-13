---
name: S8 filter colors
overview: "Align heuristic-filtered scatter colors with Figure S8c: passing points use `#2ca02c`, and grey background (excluded) points become a darker grey—without recoloring non-overlay figures like Figure 4."
todos:
  - id: darken-bg-grey
    content: Update `_PRE_PREDICATE_BG_COLOR` in plotly_helpers.py to rgb(100,100,100)
    status: completed
  - id: green-fg-overlay
    content: Add FigureS8 color constants; pass green color_discrete_sequence when should_overlay_heuristic_filter
    status: completed
  - id: sync-maps-stats
    content: Update `_heuristic_filter_color_map`, `_STATS_ANNOTATION_LABEL_COLOR`, and summary-bar heuristic_color to shared constant
    status: completed
isProject: false
---

# Heuristic filter scatter color alignment

## Current behavior

PBE scatter with heuristic overlay (Figure S8 / S9) draws two Plotly layers:

- **Background** = all constrained points → faint grey `rgb(170,170,170)` @ opacity 0.45 in [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py)
- **Foreground** = heuristic-passing only → publication black via `color_discrete_sequence=['black']` in `_helper_perform_plot_pre_post_delta_scatter`

Panel **c** already uses Matplotlib Tab10 green `#2ca02c` as `heuristic_color` in `plot_fig2_style_summary_four_bar` (~4219 in [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py)).

Overlay is enabled only when `should_overlay_heuristic_filter` is true (~5544–5547):

```5544:5547:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py
                if should_overlay_heuristic_filter:
                    passing_mask = additional_filter_predicates['is_heuristic_passing'](group_df).astype(bool)
                    plot_df = cls._copy_df_preserving_context(group_df[passing_mask])
                    active_plot_kwargs['background_data_results_df'] = group_df
```

Caller kwargs merge **after** the black defaults (`px_scatter_kwargs | kwargs.pop(...)` at ~10705–10706), so a green sequence passed from the overlay branch overrides black without changing Figure 4.

## Changes

### 1. Darker grey for all background overlay layers

In [`plotly_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) (~755):

- Change `_PRE_PREDICATE_BG_COLOR` from `'rgb(170,170,170)'` to `'rgb(100,100,100)'` (darker secondary grey; keep opacity 0.45)
- Update the nearby comment that references the old `rgba(170,170,170,0.45)` values

This automatically applies to every figure that uses `background_data_results_df` (S8, S9, and any future overlays).

### 2. Green foreground only when heuristic overlay is on

In [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) `FigureS8`:

- Add class constants next to the existing maps (~5218):
  - `HEURISTIC_PASSING_COLOR = '#2ca02c'` (same as panel c)
  - `HEURISTIC_EXCLUDED_COLOR = 'rgb(100,100,100)'` / hex `#646464` for docs/stats consistency
- Update `_heuristic_filter_color_map` and `_STATS_ANNOTATION_LABEL_COLOR` to use those constants (`True` → green, `False` → darker grey)
- In `_perform_plot_iteratively_constrained`, when `should_overlay_heuristic_filter`, merge into `active_plot_kwargs`:

```python
_fg = dict(color_discrete_sequence=[cls.HEURISTIC_PASSING_COLOR])
active_plot_kwargs['px_scatter_kwargs'] = {**(active_plot_kwargs.get('px_scatter_kwargs') or {}), **_fg}
active_plot_kwargs['hist_kwargs'] = {**(active_plot_kwargs.get('hist_kwargs') or {}), **_fg}
```

Figure 4 (`enable_heuristic_filtered_outputs: False`) stays black. Figure S3 (time-bin coloring) is unchanged.

### 3. Optional consistency in summary bars (same file)

Replace the local `heuristic_color = '#2ca02c'` in `plot_fig2_style_summary_four_bar` (~4219) and the sibling distribution helper (~5023) with `FigureS8.HEURISTIC_PASSING_COLOR` (or a shared module-level alias) so panel c and scatter stay locked to one value. No visual change for panel c.

### 4. Notebook

No notebook edits. Re-running the existing `FigureS8.main_all_figs` cell regenerates SVGs with the new colors.

## Out of scope

- VectorCraft SVG compositing scripts (consume regenerated exports)
- Changing non-overlay publication black scatters (Figure 4 Laps/PBEs without heuristic overlay)
