---
title: all relevant PyQtGraph heatmap image functions and classes
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/f0638786-77a3-43c3-80b2-2746351172ad
url: https://phohale.sourcegraph.app/deepsearch/353
created: 2026-03-01T10:46:03Z
updated: 2026-07-22T17:38:44Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis find all relevant PyQtGraph/pg-based heatmap/image (specifically to plot decoded 2D posteriors if possible) functions and classes

Based on my exploration of the codebase, here are all the relevant PyQtGraph-based heatmap/image functions and classes for plotting decoded 2D posteriors in [github.com/CommanderPho/pyPhoPlaceCellAnalysis](https://github.com/CommanderPho/pyPhoPlaceCellAnalysis):

## Core Heatmap Functions

### 1. **`visualize_heatmap_pyqtgraph`**
- **Location**: [src/pyphoplacecellanalysis/Pho2D/matplotlib/visualize_heatmap.py?L85-L169](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/matplotlib/visualize_heatmap.py?L85-L169)
- **Purpose**: Simple heatmap visualization using `pg.ImageItem` and `CustomPlotWidget`
- **Usage**: Basic 2D data visualization with optional colorbar, tick labels, and value labels

### 2. **`pyqtplot_plot_image_array`**
- **Location**: [src/pyphoplacecellanalysis/Pho2D/PyQtPlots/plot_placefields.py?L20-L180](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/plot_placefields.py?L20-L180)
- **Purpose**: Plots arrays of images (e.g., tuning curves) in a grid layout
- **Usage**: Display multiple placefield images simultaneously

### 3. **`display_all_pf_2D_pyqtgraph_binned_image_rendering`**
- **Location**: [src/pyphoplacecellanalysis/Pho2D/PyQtPlots/plot_placefields.py?L183-L344](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/plot_placefields.py?L183-L344)
- **Purpose**: High-level function for rendering all 2D placefields

## Specialized Classes & Widgets

### 4. **`BasicBinnedImageRenderingMixin` and `BasicBinnedImageRenderingWindow`**
- **Location**: [src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py?L298-L980](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py?L298-L980)
- **Purpose**: Base classes for rendering matrices of binned data as heatmaps
- **Features**:
  - Supports crosshairs and scrolling
  - `add_data()` method for adding subplots dynamically
  - `remove_data()` for removing subplots
  - Shared or individual colorbars
  - Uses `pg.ImageItem` for rendering

### 5. **`TimeSynchronizedPositionDecoderPlotter`** ⭐
- **Location**: [src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)
- **Purpose**: **Plots decoded position posteriors at a given moment in time**
- **Key Features**:
  - Uses `pg.ImageItem` to render posteriors ([line 140](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py?L140))
  - Time-synchronized updates via `on_window_changed()` slot
  - Supports `posterior_variable_to_render` property (e.g., 'p_x_given_n')
  - Includes animal trajectory plotting
  - Can add track shapes for Long/Short tracks

### 6. **`PyqtgraphTimeSynchronizedWidget`**
- **Location**: [src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/PyqtgraphTimeSynchronizedWidget.py?L23-L538](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/PyqtgraphTimeSynchronizedWidget.py?L23-L538)
- **Purpose**: Simplified PyQtGraph-based time-synchronized plotter
- **Features**: Base widget for time-synchronized plotting of decoded positions

### 7. **`DataSlicingVisualizer`** ⭐
- **Location**: [src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6621-L6750](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6621-L6750)
- **Purpose**: **Visualizes 3D data slicing for decoded 2D posteriors (P[x][y][t])**
- **Features**:
  - Uses `pg.ImageView` widget
  - Time slider for navigating through time bins
  - Can show animal's measured position marker
  - Implements `Decoded2DPosteriorTimeSyncMixin`

### 8. **`BinByBinDecodingDebugger._helper_simply_plot_posterior_in_pyqtgraph_plotitem`**
- **Location**: [src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/BinByBinDecodingDebugger.py?L330-L346](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/BinByBinDecodingDebugger.py?L330-L346)
- **Purpose**: Helper method to plot posterior in a PyQtGraph PlotItem
- **Features**: Creates `pg.ImageItem` with proper bounds and colormap

## Mixins

### 9. **`Decoded2DPosteriorTimeSyncMixin`**
- **Location**: [src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6554-L6591](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6554-L6591)
- **Purpose**: Provides time-synchronization interface for 2D posterior plots
- **Methods**: `update()`, `on_window_changed()`, `on_window_duration_changed()`

## Key Technical Details

All these use **`pg.ImageItem`** as the core PyQtGraph component for rendering 2D heatmaps/images. Common patterns:
- `pg.ImageItem(image=data)` - create the image item
- `setColorMap()` / `setLookupTable()` - apply colormaps (typically 'viridis' or 'jet')
- `setImage()` with `rect=image_bounds_extent` for proper spatial positioning
- `PlotItem.addItem(img_item)` to add to plots

## Sources

- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/matplotlib/visualize_heatmap.py@26f673d?L85-169](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/matplotlib/visualize_heatmap.py?L85-169)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py@26f673d?L298-400](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py?L298-400)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L6554-6650](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6554-6650)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/PyqtgraphTimeSynchronizedWidget.py@26f673d?L23-150](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/PyqtgraphTimeSynchronizedWidget.py?L23-150)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py@26f673d?L400-550](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/BinnedImageRenderingWindow.py?L400-550)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L6650-6750](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6650-6750)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/plot_placefields.py@26f673d?L20-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/plot_placefields.py?L20-100)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ pg.ImageItem posterior](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20pg.ImageItem%20posterior)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/PyqtgraphTimeSynchronizedWidget.py@26f673d?L150-300](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/PyqtgraphTimeSynchronizedWidget.py?L150-300)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py@26f673d?L1-300](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py?L1-300)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/BinByBinDecodingDebugger.py@26f673d?L330-400](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/BinByBinDecodingDebugger.py?L330-400)

