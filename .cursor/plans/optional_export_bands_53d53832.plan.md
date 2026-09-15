---
name: Optional export bands
overview: Add a DataFrameFilter flag (default False) so Copy/Save export figure-only by default, and only composite controls/table bands when enabled.
todos:
  - id: add-flag
    content: Add include_export_controls_and_table=False field on DataFrameFilter
    status: completed
  - id: gate-bands
    content: Gate controls/table bands in _get_export_png_bytes on that flag
    status: completed
isProject: false
---

# Optional controls/table export bands

## Goal

Copy / Save default back to **figure-only** PNG (pre-today behavior). Controls + table bands only when explicitly enabled.

## Change

In [`PhoDiba2023Paper.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py):

1. Add field next to similar export knobs (~L2755):

```python
include_export_controls_and_table: bool = non_serialized_field(default=False)
```

2. In `_get_export_png_bytes` (~L3234): always render the Plotly figure via `figure_to_pil_image` / `pio`-equivalent path; wrap the controls-band + table-band block in `if self.include_export_controls_and_table:`. When False, return PNG bytes of the figure alone (same as historical Copy/Save).

No UI toggle unless already trivial — set via attribute: `df_filter.include_export_controls_and_table = True`.

## Out of scope

- Changing polish helpers (`_format_export_control_value`, fonts)
- Notebook edits
