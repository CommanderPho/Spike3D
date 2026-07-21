# Interactive Matplotlib Mosaic Designer

Build a standalone Python script that spawns an interactive matplotlib window allowing the user to visually design a `subplot_mosaic` layout, then generates copyable Python code matching the user's conventions from [matplotlib_mosaic_example.ipynb](file:///h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/LibrariesExamples/Matplotlib/matplotlib_mosaic_example.ipynb).

## Conventions Observed in the Notebook

From the notebook, the user creates mosaics via:
```python
fig = plt.figure(figsize=(12, 10))
mosaic_layout = [
    ["ax0", "ax1", "ax2", ...],
    ["ax_main", "ax_main", ...]
]
ax_dict = fig.subplot_mosaic(
    mosaic_layout,
    height_ratios=[1, 9],
    gridspec_kw=dict(wspace=0.1, hspace=0.1)
)
```

Key conventions:
- Uses **list-of-lists** format (not ASCII art strings)
- Uses `fig.subplot_mosaic(...)` (not `plt.subplot_mosaic`)
- Explicitly sets `height_ratios` and/or `width_ratios`
- Uses `gridspec_kw` with `wspace`/`hspace`
- Axes labels follow a `"name"` pattern (e.g. `"ax0"`, `"ax_main"`)
- Empty cells use a sentinel like `"BLANK"` or `"."` with `empty_sentinel` parameter

## Design

### Technology
- Pure Python + `matplotlib` (with embedded Tk widgets via `matplotlib.backends.backend_tkagg`)
- Uses `tkinter` for the interactive UI (buttons, entry fields, scrollable code output)
- No external dependencies beyond matplotlib and tkinter (both standard in most Python environments)

### UI Layout

The application window will be divided into two major panels:

**Top Panel — Mosaic Preview (matplotlib Figure embedded in Tk)**
- Shows the current mosaic layout with labeled, colored cells
- Each cell displays its name as text
- Clicking a cell selects it (highlights border) for rename/resize operations

**Bottom Panel — Controls + Code Output (tkinter widgets)**
- **Grid Controls**: Add/remove rows/columns, set row/column count
- **Cell Controls**: Rename selected cell, merge/split cells
- **Ratio Controls**: Adjust `height_ratios` and `width_ratios` per row/col
- **Spacing Controls**: `wspace` and `hspace` sliders
- **Figure Size**: `figsize` width/height entry
- **Code Output**: A read-only `Text` widget showing the generated Python code, with a "Copy to Clipboard" button

### Core Data Model

```python
class MosaicModel:
    """Internal model representing the mosaic grid."""
    rows: int         # number of grid rows
    cols: int         # number of grid cols
    grid: list[list[str]]  # 2D array of cell labels (or empty_sentinel)
    height_ratios: list[float]
    width_ratios: list[float]
    wspace: float
    hspace: float
    figsize: tuple[float, float]
    empty_sentinel: str  # default "."
```

Cells spanning multiple grid positions share the same label string. Merging cells = painting adjacent cells with the same name.

### Interaction Flow

1. **Add Row** — appends a new row at bottom, each cell gets a unique auto-name
2. **Remove Row** — removes last row (warns if cells span into it)
3. **Add Column** — appends a new column at right
4. **Remove Column** — removes last column
5. **Select Cell** — click on a cell in the preview to select it
6. **Rename Cell** — type a new name and press Enter; updates all grid positions sharing the old name
7. **Merge Cells** — select a rectangular region (click + shift-click) and assign them the same name → creates a spanning cell
8. **Split Cell** — revert a multi-span cell back to unique individual names
9. **Adjust Ratios** — per-row and per-col ratio spinboxes
10. **Adjust Spacing** — wspace/hspace sliders
11. **Code updates live** as any change is made

### Code Generation

The generated code block will match the notebook conventions exactly:

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(W, H))
mosaic_layout = [
    ["name1", "name2", ...],
    ...
]
ax_dict = fig.subplot_mosaic(
    mosaic_layout,
    height_ratios=[...],
    width_ratios=[...],
    gridspec_kw=dict(wspace=X, hspace=Y),
)
```

## Proposed Changes

### [NEW] [mosaic_designer.py](file:///h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/LibrariesExamples/Matplotlib/mosaic_designer.py)

A single self-contained Python script (~500-700 lines) containing:
- `MosaicModel` class — data model for the grid layout
- `MosaicDesigner` class — the main Tk application with:
  - Embedded matplotlib figure for preview
  - tkinter control panel (buttons, entries, sliders)
  - Live code generation text area
- `main()` entry point

## Verification Plan

### Manual Verification
- Run the script: `python mosaic_designer.py`
- Verify the interactive window opens with a default 2×2 mosaic
- Add/remove rows and columns
- Select cells and rename them
- Merge cells into spanning regions
- Adjust height/width ratios and spacing
- Verify the generated code block updates live
- Copy the code and paste into a separate script to confirm it produces the expected layout

> [!IMPORTANT]
> This is a standalone GUI tool — automated unit tests are not practical here. Manual interaction testing is the primary verification method.
