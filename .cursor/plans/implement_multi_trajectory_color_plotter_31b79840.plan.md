---
name: Implement multi_trajectory_color_plotter
overview: Implement the `multi_trajectory_color_plotter` function using pyqtgraph to plot multiple position trajectories in a single plot with configurable rendering modes (solid colors, alpha gradients, or time-based diverging colors).
todos:
  - id: add_imports_and_validation
    content: Add pyqtgraph imports and validate input position_dfs (check for empty list and required columns)
    status: completed
  - id: create_plot_widget
    content: Create PlotWidget, configure axes labels and aspect ratio
    status: completed
    dependencies:
      - add_imports_and_validation
  - id: implement_solid_colors
    content: Implement 'solid_colors' rendering mode with unique colors per trajectory
    status: completed
    dependencies:
      - create_plot_widget
  - id: implement_alpha_gradient
    content: Implement 'alpha_gradient' mode with alpha fading from 0.25 to 0.9
    status: completed
    dependencies:
      - create_plot_widget
  - id: implement_time_diverging
    content: Implement 'time_diverging' mode with normalized time-based colormap
    status: completed
    dependencies:
      - create_plot_widget
  - id: add_return_logic
    content: Add return statement returning both PlotWidget and PlotItem based on return_widget parameter
    status: completed
    dependencies:
      - implement_solid_colors
      - implement_alpha_gradient
      - implement_time_diverging
---

# Implement multi_trajectory_color_plotter Function

## Overview

Implement the `multi_trajectory_color_plotter` function in [`PredictiveDecodingComputations.py`](pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py) to plot multiple position trajectories using pyqtgraph with support for multiple rendering modes.

## Implementation Details

### Function Signature

```python
def multi_trajectory_color_plotter(position_dfs: List[pd.DataFrame], rendering_mode: str = 'solid_colors', return_widget: bool = True) -> Tuple[pg.PlotWidget, pg.PlotItem]:
```

### Key Components

1. **Imports**: Add pyqtgraph imports at the function level (following the pattern in the file around lines 4356, 4374)

2. **Position DataFrame Structure**: 

   - Position dataframes have 'x' and 'y' columns (as seen in lines 4566-4567)
   - Optional 't' column for time-based rendering modes

3. **Rendering Modes**:

   - **'solid_colors'** (default): Each trajectory gets a unique color using `pg.intColor()` (similar to line 1819 in PendingNotebookCode.py)
   - **'alpha_gradient'**: Each trajectory uses its assigned color but with alpha gradient from 0.25 to 0.9 along the path
   - **'time_diverging'**: Uses a diverging colormap where -1.0 = start of recording, +1.0 = end of recording

4. **PyQtGraph Implementation**:

   - Create a `pg.PlotWidget()` (similar to line 4360)
   - For each trajectory, create a `pg.PlotDataItem` (similar to line 1807)
   - Use `plot_widget.plot()` or create `PlotDataItem` directly
   - For gradient modes, use `pg.PlotDataItem` with pen colors that vary along the path

5. **Color Assignment**:

   - Use `pg.intColor(i, hues=len(position_dfs))` for unique colors per trajectory
   - For alpha gradients, create pen colors with varying alpha values
   - For time diverging, normalize time values to [-1.0, 1.0] and use a diverging colormap

6. **Return Values**:

   - Return tuple: `(plot_widget, plot_item)` where `plot_item = plot_widget.getPlotItem()`
   - If `return_widget=False`, return only the PlotItem

### Implementation Steps

1. Add function body with pyqtgraph imports
2. Validate input: check that position_dfs is not empty and each df has 'x' and 'y' columns
3. Create PlotWidget and configure axes labels ('X Position', 'Y Position')
4. Implement color generation based on rendering_mode
5. Loop through position_dfs and plot each trajectory:

   - Extract x, y coordinates
   - For solid_colors: plot with unique color
   - For alpha_gradient: create segments with varying alpha
   - For time_diverging: compute normalized time values and apply colormap

6. Set aspect ratio and auto-range
7. Return PlotWidget and PlotItem

### Notes

- Follow the existing code style: single-line function signatures, two blank lines between methods
- Handle missing 't' column gracefully for time-based modes
- Use pyqtgraph's color utilities (`pg.intColor`, `pg.mkPen`, `pg.mkBrush`) for consistency
- For time_diverging mode, compute global time range across all trajectories if 't' column exists