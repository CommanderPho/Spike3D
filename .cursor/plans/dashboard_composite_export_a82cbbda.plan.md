---
name: Dashboard composite export
overview: Minimally extend DataFrameFilter Copy/Save so both export a composite PNG (filter state + Plotly figure + stats table) by reusing existing export helpers instead of new renderers.
todos:
  - id: get-png-bytes
    content: Add thin _get_export_png_bytes() reusing figure_to_pil_image, get_df_filter_active_constraint_dict, add_boxed_adjacent_label, table_widget.data
    status: completed
  - id: wire-buttons
    content: Point Copy handler and Solara Save at _get_export_png_bytes with minimal signature tweak to _build_solera_file_download_widget
    status: completed
isProject: false
---

# Dashboard composite Copy/Save export (minimal)

## Goal

Copy / Save export **filter controls + Plotly figure + filter-stats table** (not figure-only). Exclude hover preview and `output_widget`.

Constraint: **minimal edits**; reuse extant helpers; fix them only if needed for multiline labels.

## Reuse (do not reinvent)

| Need | Existing code |
|------|----------------|
| Filter control state | [`DataFrameFilter.get_df_filter_active_constraint_dict()`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py) (~L3398); also append plot df/variable + checked predicates |
| Figure → PIL | [`figure_to_pil_image`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/plotting/media_output_helpers.py) (~L1752) |
| Text band above/below image | [`ImageOperationsAndEffects.add_boxed_adjacent_label`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/plotting/media_output_helpers.py) (~L327) |
| Stats table data | `self.table_widget.data` (already updated at ~L4418); format with `df.to_string()` |
| Clipboard / download UX | Keep existing Copy JS path and Solara `FileDownload` — only swap the PNG byte source |

Do **not** change [`add_copy_save_action_buttons`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/plotly/Extensions/plotly_helpers.py). Do **not** add matplotlib panel renderers or `vertical_image_stack` unless `add_boxed_adjacent_label` cannot handle the needed text.

## Changes (only these)

### 1. One method on `DataFrameFilter`

`_get_export_png_bytes(self) -> bytes` in [`PhoDiba2023Paper.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py):

1. `img = figure_to_pil_image(self.figure_widget, format='png', ...)` (preserve width/height kwargs from current Copy handler)
2. Build a short multiline control summary from `get_df_filter_active_constraint_dict()` plus plot-df / plot-variable / predicate selector values
3. `img = ImageOperationsAndEffects.add_boxed_adjacent_label(img, controls_text, image_edge='top', ...)` (light bg, dark text)
4. `table_text = self.table_widget.data.to_string()` (fallback: `filtered_size_info_df` if data missing)
5. `img = ImageOperationsAndEffects.add_boxed_adjacent_label(img, table_text, image_edge='bottom', ...)`
6. Save PIL → PNG bytes

### 2. Wire both buttons to that method

- Copy (`_subfn_on_copy_button_click` ~L3228): replace `pio.to_image(...)` with `_get_export_png_bytes()`; leave clipboard JS untouched
- `_build_solera_file_download_widget` (~L2620): add optional `get_png_bytes` callable; default remains `lambda: pio.to_image(fig, ...)` for any other callers; from `_setup_widgets_buttons` pass `get_png_bytes=self._get_export_png_bytes`

### 3. Fix-as-needed only

If `add_boxed_adjacent_label` breaks on multiline (`\n`) text (PIL `.text` vs `.multiline_text` / `textsize`), make the **smallest** fix in that helper so stacked lines work for top/bottom bands. No broader refactor.

## Out of scope

- DOM / html2canvas
- Pixel-perfect widget chrome
- New matplotlib table/control renderers
- Changes to standalone plotly action-button helper
