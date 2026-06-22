---
name: Fix Bapun epoch interval colors and labels
overview: Fix the uniform/incorrect pen colors and the lost per-epoch labels for Bapun paradigm intervals, with surgical changes that leave the behavior of currently-working sessions and single-label interval series (PBEs, Laps, SessionEpochs) unchanged.
todos:
  - id: fix-pen-color
    content: In build_bapun_proper_epoch_intervals (PendingNotebookCode.py ~line 4900), derive pen_color from per-epoch lap_color instead of the constant lap_accent_color.
    status: completed
  - id: fix-config-extraction
    content: In init_configs_list_from_interval_datasource_df (epochs_plotting_mixins.py ~189-197), serialize pen/brush via ColorDataframeColumnHelpers.QPen_to_tuple/QBrush_to_tuple (with isinstance guards) before init_from_visualization_dataframe_row so the per-row label good-path no longer crashes; leave the fallback branch unchanged.
    status: completed
  - id: validate
    content: "Re-run on RatK D3 TwoNovel: verify per-epoch labels and distinct pen colors in an_interval_ds.df and the config widget; spot-check a working session plus PBEs/Laps for no regression."
    status: completed
isProject: false
---

# Fix Bapun epoch interval colors and labels

Two independent root causes; two targeted fixes. Both are designed to be non-breaking for sessions/series that already work.

## Problem 2 (pen_color all identical) - isolated to the Bapun helpers

In [build_bapun_all_epochs_df](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) `lap_accent_color` is a constant `'#FFFFFF'` for every row, and [build_bapun_proper_epoch_intervals](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) derives `pen_color` from it, so every pen is white.

Fix (line ~4900): derive the pen from the per-epoch `lap_color` (same source the brush already uses), keeping the 0.8 alpha for the pen:

```python
curr_paradigm_df['pen_color'] = [inline_mkColor(c, 0.8) for c in curr_paradigm_df['lap_color'].tolist()]
```

- This is fully contained in the Bapun-only helper, so it cannot affect any other session or series.
- Optional variant if a darker outline is preferred: keep `lap_accent_color` but populate it per-epoch in `build_bapun_all_epochs_df` (e.g. a darkened `lap_color`) instead of the constant white. Default to the simple `lap_color` version unless told otherwise.

## Problem 1 (label all 'custom_paradigm') - fix the root crash, preserve fallback

The labels are correct inside `build_bapun_proper_epoch_intervals`; they get overwritten by the config-widget round-trip:

1. [init_configs_list_from_interval_datasource_df](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\PhoPositionalData\plotting\mixins\epochs_plotting_mixins.py) (the "good" path) passes the raw `df['pen']`/`df['brush']` (live `QPen`/`QBrush`) into `init_from_visualization_dataframe_row`, which calls `len(pen_tuple)` -> `object of type 'QPen' has no len()` -> falls into the `except`.
2. The fallback builds configs named after the datasource (`'custom_paradigm'`) and drops the per-row labels.
3. Applying those configs back runs `_update_df_visualization_columns`, which writes the config `name` into `df['label']` ([Specific2DRenderTimeEpochs.py:77-82](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\Specific2DRenderTimeEpochs.py)), so all rows become `'custom_paradigm'` (and the change is self-reinforcing via the uniqueness assert).

Fix: in the "good" path of `init_configs_list_from_interval_datasource_df`, serialize pens/brushes to tuples before calling `init_from_visualization_dataframe_row`, so the per-row `label` naming actually succeeds. `QPen_to_tuple` -> `QPenTuple(color, width)` (len 2) and `QBrush_to_tuple` -> `QBrushTuple(color)` (len 1) are already accepted by `init_from_visualization_dataframe_row`.

Sketch (lines ~189-197):

```python
from pyphocorehelpers.gui.Qt.color_helpers import ColorDataframeColumnHelpers
...
a_serializable_df = a_ds.df.copy()
assert np.all(np.isin(['series_vertical_offset','series_height','pen','brush'], a_serializable_df.columns))
label_names = a_serializable_df['label'].to_list()
unique_label_names = np.unique(label_names)
assert len(unique_label_names) == len(a_serializable_df), f"..."
pen_tuples   = [ColorDataframeColumnHelpers.QPen_to_tuple(p)   if isinstance(p, QPen)   else p for p in a_serializable_df['pen']]
brush_tuples = [ColorDataframeColumnHelpers.QBrush_to_tuple(b) if isinstance(b, QBrush) else b for b in a_serializable_df['brush']]
out_list = [cls.init_from_visualization_dataframe_row(a_name, y_location, height, a_pen, a_brush)
            for a_name, y_location, height, a_pen, a_brush
            in zip(label_names, a_serializable_df['series_vertical_offset'], a_serializable_df['series_height'], pen_tuples, brush_tuples)]
return out_list
```

### Why this does not break currently-working sessions
- Single-label series (PBEs, Laps, SessionEpochs): the `len(unique_label_names) == len(df)` assert still fails (1 != N), so they still take the existing fallback path unchanged.
- Multi-label paradigm with proper distinct labels: the good path now succeeds and produces correctly-named per-epoch configs, so the apply round-trip writes the real labels back (idempotent) instead of `'custom_paradigm'`.
- The `isinstance(..., QPen/QBrush)` guards pass through any already-serialized tuples, so a session whose good path already worked is unaffected.
- The fallback branch and `_update_df_visualization_columns` are left untouched, minimizing blast radius.

## Validation
- Re-run `build_bapun_proper_epoch_intervals` on the RatK D3 TwoNovel session and confirm `an_interval_ds.df['label']` is `['pre','maze1','post1','maze2','post2']` and `pen_color` varies per epoch.
- Confirm the `EpochIntervalsVisualConfigs` widget shows the five distinct paradigm names instead of five `custom_paradigm` entries.
- Sanity check a session that currently works (and PBEs/Laps rendering) to confirm no regression.