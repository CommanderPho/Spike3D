---
title: interactive lap debugger code
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/a13be301-a51f-48a5-8449-926e0641c394
url: https://phohale.sourcegraph.app/deepsearch/195
created: 2025-10-21T17:01:06Z
updated: 2026-07-22T17:44:53Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis where is code for displaying an interactive lap debugger/diagnoser plot and UI?

The interactive lap debugger/diagnoser plot and UI code is primarily located in the [`EpochsEditorItem.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py) file. Here's a comprehensive overview:

## Main Components

### **Primary Class: `EpochsEditor`**
Located at: [`src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py)

This is the core class that provides an interactive GUI for debugging and modifying lap epochs using PyQtGraph.

### **Key Features:**

1. **Interactive Visualization**: 
   - Displays position data (`x_smooth`)
   - Optional velocity (`velocity_x_smooth`) and acceleration plots
   - Color-coded lap regions (green for LR direction, purple for RL direction)

2. **User Interactions**:
   - **Drag to resize**: Lap boundaries can be dragged to adjust start/stop times
   - **Click to toggle**: Select/deselect laps (toggles `is_included` flag)
   - **Right-click context menu**: Toggle lap direction or remove laps
   - **Double-click**: Add new lap epochs

3. **Visual Elements**:
   - Track boundaries as dotted horizontal lines
   - Grid bin bounds as thick horizontal lines
   - Zero-crossing reference lines
   - Customizable epoch labels and colors

### **Main Entry Points:**

1. **`EpochsEditor.init_laps_diagnoser()`** - Initialize from position and laps dataframes
2. **`EpochsEditor.init_from_session()`** - Initialize directly from a session object
3. **`EpochsEditor.perform_plot_laps_diagnoser()`** - Create the plotting interface

### **Usage Example:**

```python
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsWidgets.EpochsEditorItem import EpochsEditor

# Initialize from session
epochs_editor = EpochsEditor.init_from_session(
    global_session, 
    include_velocity=True, 
    include_accel=False
)

# Or initialize from dataframes
epochs_editor = EpochsEditor.init_laps_diagnoser(
    pos_df, 
    curr_laps_df, 
    include_velocity=True, 
    include_accel=False
)
```

### **Related Components:**

- **Custom ViewBox**: [`CustomViewBox`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py?L26-L42) for handling mouse interactions
- **Linear Region Items**: Uses `CustomLinearRegionItem` for draggable lap boundaries
- **Event Callbacks**: Handles region updates and selection changes

### **Integration Points:**

The lap debugger is used in the broader pipeline through:
- [`DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L7985-L8031) for directional laps overview
- Various display functions that need interactive lap editing capabilities

The diagram above shows the complete architecture and data flow of the interactive lap debugger system, from session data input through the interactive GUI components to the final user interactions and callbacks.

## Sources

- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ lap debug diagnos interactive plot UI](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20lap%20debug%20diagnos%20interactive%20plot%20UI)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ lap AND (debug OR diagnos OR interactive)](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20lap%20AND%20%28debug%20OR%20diagnos%20OR%20interactive%29)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py@26f673d](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/GraphicsWidgets/EpochsEditorItem.py)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "EpochsEditor" OR "perform_plot_laps_diagnoser" OR "init_laps_diagnoser"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22EpochsEditor%22%20OR%20%22perform_plot_laps_diagnoser%22%20OR%20%22init_laps_diagnoser%22)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ lap overlap debugger interactive plot UI widget](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20lap%20overlap%20debugger%20interactive%20plot%20UI%20widget)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:^notebooks lap debugger diagnoser EpochsEditor](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:%5Enotebooks%20lap%20debugger%20diagnoser%20EpochsEditor)

## Suggested Follow-ups

- How does the EpochsEditor class handle the underlying data updates when a user interacts with the plot?
- Can you provide a more detailed example of how to integrate the EpochsEditor into an existing PyQt application?

