---
name: Heuristic grey series export
overview: "When `enable_heuristic_filtered_outputs` is True (Figure S8/S9 PBEs), reuse the existing `background_data_results_df` grey layer: unconstrained-by-heuristic rows as grey background, heuristic-passing rows as the normal foreground — replacing the current False/True concat overlay without changing Figure 4/S3 or stats."
todos:
  - id: heuristic-bg-export
    content: "In _perform_plot_iteratively_constrained: pass background_data_results_df=group_df and plot heuristic-passing foreground; remove heuristic color/overlay concat"
    status: completed
isProject: false
---

# Grey pre-heuristic series for non-interactive Figure S8/S9

## Current behavior

When `enable_heuristic_filtered_outputs=True`, PBEs plots set `should_overlay_heuristic_filter=True` in [`FigureS8.main`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) (~4017). That path:

1. Builds a stacked df via `_overlay_heuristic_filter_rows` (all rows labeled `False` + passing labeled `True`)
2. Colors by `heuristic_filter` with `_heuristic_filter_color_map` (grey / black)

## Desired behavior

Reuse the interactive-path mechanism already in [`plotly_pre_post_delta_scatter`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) (`background_data_results_df`):

- **Grey** = constrained `group_df` (no heuristic filter)
- **Foreground** = heuristic-passing subset only (normal publication styling)

Figure 4 / S3 (`enable_heuristic_filtered_outputs=False`) unchanged.

## Minimal change — only `_perform_plot_iteratively_constrained`

**File:** [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py)

1. **Stop setting heuristic color kwargs** when `should_overlay_heuristic_filter` (~3633–3638). Keep default publication `_plot_kwargs` (black series).

2. **In `_export_stage`** (~3668–3687), replace overlay concat with:

```python
if should_overlay_heuristic_filter:
    passing_mask = additional_filter_predicates['is_heuristic_passing'](group_df).astype(bool)
    plot_df = cls._copy_df_preserving_context(group_df[passing_mask])
    active_plot_kwargs['background_data_results_df'] = group_df
else:
    plot_df = group_df
```

`_plot_formatted_pre_post_scatter` already forwards `**plot_kwargs` into `_perform_plot_pre_post_delta_scatter`, which already accepts `background_data_results_df`.

3. **Legend:** drop the heuristic-only `showlegend`/`legend_title_text` special case. Use the same rule as non-overlay plots (`showlegend` from `color_by_time_bin_size` / `show_scatter_legend` only).

4. **Stats:** leave the existing `group_df` / `passing_mask` False/True annotation path (~3701–3713) as-is — still compares all vs heuristic-passing.

5. **Debug (`store_debug_values`):** store `plot_df` as the heuristic-passing foreground (and keep `heuristic_passing_df` print path ~3779–3785). Stop calling `_overlay_heuristic_filter_rows` from this export path. Leave `_overlay_heuristic_filter_rows` / `_heuristic_filter_color_map` defined (unused by this path) to avoid a broader cleanup.

No notebook edits. No new helpers unless a one-liner in `_export_stage` is insufficient (it is sufficient).
