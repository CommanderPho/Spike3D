---
name: Export band polish
overview: "Polish DataFrameFilter composite export bands: human-readable parameter values, Plotly-matching Arial/DejaVu font at readable size, and monospace-aligned table text."
todos:
  - id: format-values
    content: Humanize tuple/list control values in _get_export_png_bytes
    status: completed
  - id: fonts-sizes
    content: Use Arial/DejaVu size 14 for controls; FreeMono size 12 for table band
    status: completed
isProject: false
---

# Polish composite export text bands

## Problem

[`_get_export_png_bytes`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py) (~L3213) produces top/bottom bands that look wrong next to the Plotly figure:

- **Font**: default label font is `ndastroneer.ttf`; with `relative_font_size=0.012` size collapses to ~8px, so bands look like a tiny system/default font vs Plotly’s **Arial**.
- **Values**: raw `str(v)` dumps Python tuples (`('dropped',)`, `(0.075,)`, long predicate tuples).
- **Table**: `DataFrame.to_string()` under a proportional font misaligns columns.

The pipe-separated footer *inside* the Plotly canvas is pre-existing figure annotation content (not from the new bands); leave it alone.

## Fix (minimal, only in `_get_export_png_bytes`)

### 1. Human-readable value formatting

Add a tiny local helper (or inline) used when building control rows:

- `tuple`/`list` of length 1 → `str(item)` (so `time_bin_size: 0.075`, `masked_time_bin_fill_type: dropped`)
- longer sequences → `", ".join(...)` (so predicates read as `is_track_body, trained_compute_epochs_widget, ...`)
- scalars unchanged

Apply to constraint-dict values and to `filter_predicates` / plot selectors.

### 2. Match Plotly typography for the controls band

Pass explicit kwargs into `add_boxed_adjacent_label` for the **top** band:

- `font`: resolve **Arial** (Windows `arial.ttf` / `C:/Windows/Fonts/arial.ttf`) when present; else existing bundled [`DejaVuSans.ttf`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/Resources/fonts/DejaVuSans/DejaVuSans.ttf) via `ImageHelpers.get_font` naming (`DejaVuSans.ttf`)
- `font_size=14` (absolute, not `relative_font_size=0.012`)
- keep light bg / dark text; optional `spacing=4` via text kwargs for multiline breathing room

### 3. Monospace-aligned table band

For the **bottom** band:

- format with `table_df.to_string()` as today (keeps column structure)
- pass `font='FreeMono.ttf'` (bundled in [`Resources/fonts`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers/Resources/fonts/FreeMono.ttf)) so columns line up
- `font_size=12` (readable, slightly smaller than controls)

No new matplotlib table renderer; no changes to `add_copy_save_action_buttons` or the Plotly figure footer.

## Files

- Only [`PhoDiba2023Paper.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py) `_get_export_png_bytes` (plus a 5-line format helper next to it if clearer)
