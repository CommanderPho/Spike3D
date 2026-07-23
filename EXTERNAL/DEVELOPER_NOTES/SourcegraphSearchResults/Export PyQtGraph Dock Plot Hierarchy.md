---
title: Export PyQtGraph Dock Plot Hierarchy
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/0ce7a928-0c96-40f9-9075-20a4c5ff7769
url: https://phohale.sourcegraph.app/deepsearch/359
created: 2026-04-01T08:33:40Z
updated: 2026-04-01T08:34:50Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/CommanderPho/pyPhoCoreHelpers Find existing code to export an entire visual hierarchy of PyQtGraph plots nested in pg.Dock items

Based on my thorough exploration of your codebase, I found the main export functionality for PyQtGraph plots. Here's what exists:

## Existing Code for Exporting PyQtGraph Plots in Dock Hierarchies

### 1. **Single Plot Export Function**

The main function for exporting individual PyQtGraph elements is [`export_pyqtgraph_plot`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py?L96-L156):

```python
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot

# Export a single PlotItem, GraphicsLayoutWidget, or other pyqtgraph item
export_pyqtgraph_plot(graphics_item, savepath='fileName.png')  # PNG
export_pyqtgraph_plot(graphics_item, savepath='fileName.svg')  # SVG
```

**Supported inputs:**
- `PlotItem`
- `GraphicsLayoutWidget`
- `GraphicsView`
- Any pyqtgraph scene item

### 2. **Export All Plots in a BinnedImageRenderingWindow**

For windows with multiple plots, there's [`export_all_plots`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py?L603-L632):

```python
# From BasicBinnedImageRenderingMixin
out_figs_dict = window.export_all_plots(curr_active_pipeline)
out_figs_paths = [v[0][0] for v in list(out_figs_dict.values())]
```

This iterates through all plots in the window and exports each one individually.

### 3. **Accessing Docks Recursively**

While I didn't find a complete "export entire hierarchy" function, the code has utilities to traverse dock structures:

**Find all dock items:**
```python
# Get all docks recursively
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from PyQt5 import QtCore

curr_display_dock_items = displayDockArea.findChildren(
    Dock, 
    QtCore.QRegExp(".*"), 
    QtCore.Qt.FindChildrenRecursively
)
```

**Get flat list of dock identifiers:**
```python
# From DynamicDockDisplayAreaContentMixin
included_track_dock_identifiers = active_2d_plot.dock_manager_widget.get_leaf_only_flat_dock_identifiers_list()
```

### 4. **No Complete Hierarchy Export Function Found**

I **did not find** an existing function that automatically exports an entire visual hierarchy of PyQtGraph plots nested in `pg.Dock` items. You would need to create one by combining the above pieces.

## Recommended Approach to Build This Functionality

Here's how you could create a hierarchical export function:

```python
from pathlib import Path
from typing import List, Dict
from pyphoplacecellanalysis.External.pyqtgraph.dockarea.Dock import Dock
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import export_pyqtgraph_plot
from PyQt5 import QtCore

def export_dock_hierarchy(
    dock_area_window, 
    output_folder: Path, 
    export_format: str = '.png',
    **export_kwargs
) -> Dict[str, Path]:
    """
    Export all PyQtGraph plots nested in pg.Dock items recursively.
    
    Args:
        dock_area_window: PhoDockAreaContainingWindow or similar with .displayDockArea
        output_folder: Directory to save exports
        export_format: '.png' or '.svg'
        **export_kwargs: Additional args passed to export_pyqtgraph_plot (e.g., background, width)
    
    Returns:
        Dictionary mapping dock identifiers to saved file paths
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    exported_paths = {}
    
    # Get all docks recursively
    all_docks = dock_area_window.displayDockArea.findChildren(
        Dock, 
        QtCore.QRegExp(".*"), 
        QtCore.Qt.FindChildrenRecursively
    )
    
    for dock in all_docks:
        dock_id = dock.name()
        
        # Get widgets in this dock
        dock_widgets = dock.widgets
        
        for i, widget in enumerate(dock_widgets):
            # Try to export if it's a pyqtgraph item
            try:
                # Determine what type of pyqtgraph object this is
                exportable_item = None
                
                if hasattr(widget, 'plotItem'):
                    # It's a PlotWidget
                    exportable_item = widget.plotItem
                elif hasattr(widget, 'scene'):
                    # It's a GraphicsLayoutWidget or similar
                    exportable_item = widget.scene()
                elif hasattr(widget, 'getViewBox'):
                    # Direct PlotItem
                    exportable_item = widget
                
                if exportable_item is not None:
                    # Build output filename
                    suffix = f'_{i}' if len(dock_widgets) > 1 else ''
                    safe_dock_id = dock_id.replace('/', '_').replace('\\', '_')
                    output_path = output_folder / f'{safe_dock_id}{suffix}{export_format}'
                    
                    # Export
                    export_pyqtgraph_plot(
                        exportable_item, 
                        savepath=output_path,
                        progress_print=True,
                        **export_kwargs
                    )
                    
                    exported_paths[f'{dock_id}{suffix}'] = output_path
                    
            except Exception as e:
                print(f'Could not export dock "{dock_id}": {e}')
                continue
    
    return exported_paths
```

**Usage:**
```python
exported_paths = export_dock_hierarchy(
    root_dockAreaWindow,
    output_folder=Path('output/exports'),
    export_format='.png',
    background=pg.mkColor(0, 0, 0, 0),  # transparent background
    width=4096
)
```

This function will recursively find all docks and export any PyQtGraph plots they contain.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis export pyqtgraph dock plot hierarchy](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20export%20pyqtgraph%20dock%20plot%20hierarchy)
- [repo:github.com/CommanderPho/pyPhoCoreHelpers export pyqtgraph dock plot hierarchy](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20export%20pyqtgraph%20dock%20plot%20hierarchy)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis save render pyqtgraph image export](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20save%20render%20pyqtgraph%20image%20export)
- [repo:github.com/CommanderPho/pyPhoCoreHelpers save render pyqtgraph image export](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoCoreHelpers%20save%20render%20pyqtgraph%20image%20export)
- [repo:github.com/CommanderPho DockArea export OR save](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%20DockArea%20export%20OR%20save)
- [repo:github.com/CommanderPho DockAreaWrapper export](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%20DockAreaWrapper%20export)
- [repo:github.com/CommanderPho PhoDockAreaContainingWindow export](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%20PhoDockAreaContainingWindow%20export)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis dock area hierarchy recursive export](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20dock%20area%20hierarchy%20recursive%20export)
- [repo:github.com/CommanderPho displayDockArea saveState](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%20displayDockArea%20saveState)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py@26f673d?L96-180](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py?L96-180)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py@26f673d?L1110-1250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py?L1110-1250)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py@26f673d?L600-670](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py?L600-670)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis iterate traverse all docks children widgets](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20iterate%20traverse%20all%20docks%20children%20widgets)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/DockingWidgets/DynamicDockDisplayAreaContent.py@26f673d?L750-850](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/DockingWidgets/DynamicDockDisplayAreaContent.py?L750-850)

