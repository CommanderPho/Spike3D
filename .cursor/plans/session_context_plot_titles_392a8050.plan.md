---
name: Session context plot titles
overview: Add the pipeline’s session identifier to the PyQtGraph window title and a new top-row layout label (suptitle), and adjust row stretch factors so heatmap rows still expand correctly after inserting one row.
todos:
  - id: session-string
    content: Add _session_uid via get_session_context().get_description_as_session_global_uid()
    status: completed
  - id: win-title-suptitle
    content: Prefix GraphicsLayoutWidget title; add row-0 addLabel suptitle; shift banner row to 1; curr_row=2
    status: completed
  - id: stretch-loop
    content: Update setRowStretchFactor loop + comment for new row indexing
    status: completed
isProject: false
---

# Session context on occupancy comparison plot

## Context

[`plot_decoded_and_measured_occupancies`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) builds a [`pg.GraphicsLayoutWidget`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py) with:

- **Row 0:** empty `(0,0)` + per-column “Directed / Sprinkle” banners
- **Row 1+:** alternating subtitle rows and viewbox rows from `_subfn_add_single_row`

Elsewhere in this repo, the canonical human-readable session key is:

```python
curr_active_pipeline.get_session_context().get_description_as_session_global_uid()
```

(equivalent to `get_description(separator="|", include_property_names=False)` on [`IdentifyingContext`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/utils/result_context.py)). That type is **already imported** in `PendingNotebookCode.py`; no new imports.

## Implementation (minimal edits in one function)

1. **Resolve session string once** near the start of the function body (after `decoder_names` / `masked_container` handling so `decoder_names` exists for `colspan`):

   - `_session_uid = curr_active_pipeline.get_session_context().get_description_as_session_global_uid()`

2. **Window title:** extend the existing `GraphicsLayoutWidget(title=...)` string to prefix (or embed) `_session_uid`, e.g.  
   `f"{_session_uid} — BinnedOccupancyComparisons — columns: ..."`  
   keeping the rest of the current description unchanged.

3. **Suptitle (layout):** insert **new row 0** spanning the grid width:

   - `win.addLabel(text=_session_uid, row=0, col=0, colspan=1 + 2 * len(decoder_names), ...)`
   - Use modest typography (e.g. `size='12pt'`, `bold=True`) so it reads as a suptitle without overwhelming the column banners.

4. **Shift existing header row down by one:**

   - Move the current empty corner + column banner labels from `row=0` to `row=1` (same `col` / `colspan` as today).

5. **Set `curr_row` to `2`** (was `1`) before the first `_subfn_add_single_row` call.

6. **Fix row stretch factors** at the bottom of the function. The current rule (`row 0` and odd rows fixed, even rows ≥2 stretched) assumes the **column banners** are on row 0. After insertion:

   - Row **0:** suptitle — fixed  
   - Row **1:** column banners — fixed  
   - Rows **≥2:** even = subtitle rows (fixed), odd = viewbox rows (stretched)

   Replace the loop condition with the equivalent of: **stretch 0** if `_r <= 1` or `_r % 2 == 0`, else **stretch 1** (for `_r >= 2`). Update the nearby comment to describe rows 0–1 + the new parity pattern.

## Files touched

- Only [`PendingNotebookCode.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py), inside `plot_decoded_and_measured_occupancies` (~lines 523–656).

## Verification

- Run any notebook/caller that invokes this method and confirm: window title bar shows session UID; top of the grid shows the same UID; column headers and heatmaps still align; vertical space still goes to viewbox rows, not label rows.
