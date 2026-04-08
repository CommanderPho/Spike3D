---
name: Rightmost-only colorbar
overview: Conditionally create and add `pg.ColorBarItem` only for the final column in `BinnedOccupancyComparisons._subfn_add_single_row`, since each row already uses one shared scale; left/middle columns keep images only.
todos:
  - id: gate-cbar
    content: Wrap ColorBarItem create/setImageItem/addItem in `col_idx == len(column_data) - 1` in _subfn_add_single_row; extend docstring.
    status: completed
isProject: false
---

# Single shared colorbar per row (rightmost column)

## Target

`[pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py)` — nested helper `_subfn_add_single_row` inside `BinnedOccupancyComparisons.plot_decoded_and_measured_occupancies` (lines ~394–409).

## Current behavior

For each `(image_data, title)` in `column_data`, the code adds a `ViewBox` + `ImageItem` at column `col0` and a `ColorBarItem` at `col0 + 1`, with identical `(vmin, vmax)` per row — redundant colorbars.

## Change

1. **Gate the colorbar** — After `vb.setAspectLocked(True)`, only when `col_idx == len(column_data) - 1`:
  - Instantiate `pg.ColorBarItem(...)`, call `cbar.setImageItem(img_item)`, and `win.addItem(cbar, row=(curr_row + 1), col=col0 + 1)`.
2. **Docstring** — One line in the helper docstring noting that the colorbar appears once per row (rightmost panel), matching shared normalization.

No other call sites need updates; all rows go through this helper.

## Layout note

Middle columns will leave grid cell `(curr_row + 1, col0 + 1)` empty. `GraphicsLayoutWidget` typically leaves that slot minimal; if you later want panels to visually widen, that would be a separate layout tweak (e.g. spanning the ViewBox when not last). Not required for “hide duplicate colorbars.”

## Verification

Run any notebook/script that calls `plot_decoded_and_measured_occupancies` and confirm four rows each show **one** colorbar (aligned with the rightmost Directed/Sprinkle panel).