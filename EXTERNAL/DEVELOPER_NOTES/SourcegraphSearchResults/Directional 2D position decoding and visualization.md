---
title: Directional 2D position decoding and visualization
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/517017b7-1095-4085-b2cb-d63712fc6261
url: https://phohale.sourcegraph.app/deepsearch/234
created: 2025-10-30T21:12:14Z
updated: 2026-07-22T17:41:33Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis find all code, sorted in order of creation (newest to oldest) related to angular/directional 2D position decoding and visualization.

Now let me create a comprehensive summary by examining the commit history and organizing the findings:

Based on my comprehensive search of the pyPhoPlaceCellAnalysis repository, I've found extensive code related to angular/directional 2D position decoding and visualization. Here's a summary of the relevant code, sorted by creation date (newest to oldest):

## **Most Recent Code (2025)**

### **1. Directional Merged Placefield Decoded Stacked Epoch Slices (2025-05-30)**
- **File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py)
- **Function**: `_display_directional_merged_pf_decoded_stacked_epoch_slices`
- **Purpose**: Exports all session's decoded laps and ripples as stacked epoch visualizations
- **Commits**: d573eff, 9e15157, 351007e, 8ad276d

### **2. Directional Pseudo2D Decoder Results (2025-06-13)**
- **File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py)
- **Class**: `DirectionalPseudo2DDecodersResult`
- **Function**: `split_pseudo2D_continuous_result_to_1D_continuous_result`
- **Commits**: e5965b4, 5ee1654

### **3. Track ID Weighted Position Posterior with Multi-Color Overlay (2025-05-16)**
- **File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py)
- **Function**: `_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay`
- **Commits**: 4c9a04d, 8d7d664

### **4. Directional Track Template 1D Placefields (2025-06-27)**
- **File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py)
- **Function**: `_display_directional_track_template_pf1Ds`
- **Commits**: 42a282c

## **Core Decoding Infrastructure (2024-2025)**

### **5. Directional Decoders Continuously Decoded Result (2025-02-26)**
- **File**: [`src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py)
- **Class**: `DirectionalDecodersContinuouslyDecodedResult`
- **Purpose**: Container for caching continuous decoding results across multiple decoders and time bin sizes
- **Commits**: fb3c590

### **6. Angular/Directional Visualization Functions (2025)**
- **File**: [`src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py)
- **Functions**:
  - `plot_directional_occupancy` - 2D heatmap for specific head direction bins
  - `plot_spatial_angular_distributions` - Angular distribution plotting
  - Angular data visualization with circular representations

### **7. Time-Synchronized Position Decoder Plotter (2025)**
- **File**: [`src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py)
- **Class**: `TimeSynchronizedPositionDecoderPlotter`
- **Purpose**: Real-time visualization of decoded position posteriors
- **Commits**: 730517f, af42669

## **2D Position Decoding Visualization (2024-2025)**

### **8. Decoder Result Display Classes (2025-02-20)**
- **File**: [`src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py)
- **Class**: `DecoderResultDisplayingPlot2D`
- **Purpose**: Displays 2D position decoder results

### **9. Stacked Epoch Slices Visualization**
- **File**: [`src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py)
- **Purpose**: Vertically stacked visualization of decoded position epochs
- **Features**: Support for directional decoding visualization

### **10. Directional Template Debugger (2024)**
- **File**: [`src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/TemplateDebugger.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/TemplateDebugger.py)
- **Function**: `_display_directional_template_debugger`
- **Purpose**: Interactive debugging of directional placefield templates

## **Key Features Summary:**

1. **Angular Position Decoding**: Comprehensive support for directional/angular position decoding using pseudo-2D decoders
2. **Real-time Visualization**: Time-synchronized plotting of decoded position posteriors
3. **Multi-Decoder Support**: Handling of multiple directional decoders (Long/Short, LR/RL)
4. **Export Capabilities**: Extensive image export functionality for decoded posteriors
5. **Interactive Debugging**: Tools for debugging and validating directional decoding results
6. **Marginal Analysis**: Support for track identity and directional marginals
7. **Circular Visualizations**: Angular data representation with circular heatmaps

The codebase shows active development with the most recent commits from 2025, indicating this is current, actively maintained code for angular/directional 2D position decoding and visualization.

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ angular directional position decoding](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20angular%20directional%20position%20decoding)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ 2D position decode visualization](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%202D%20position%20decode%20visualization)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ angular OR directional OR decode OR decoding](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20angular%20OR%20directional%20OR%20decode%20OR%20decoding)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "angular" OR "directional" OR "theta" OR "heading"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22angular%22%20OR%20%22directional%22%20OR%20%22theta%22%20OR%20%22heading%22)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ visualization plot 2D position decoder](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20visualization%20plot%202D%20position%20decoder)
- [type:commit (message:angular OR message:directional OR message:2D OR message:position OR message:decode OR message:visualization OR message:plot) repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis count:50](/search?q=type:commit%20%28message:angular%20OR%20message:directional%20OR%20message:2D%20OR%20message:position%20OR%20message:decode%20OR%20message:visualization%20OR%20message:plot%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20count:50)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:.*visualization.* OR file:.*plot.* OR file:.*display.*](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:.%2Avisualization.%2A%20OR%20file:.%2Aplot.%2A%20OR%20file:.%2Adisplay.%2A)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@26f673d?L1-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1-100)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py@26f673d?L1-50](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/TimeSynchronizedPositionDecoderPlotter.py?L1-50)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py@26f673d?L120-150](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Analysis/Decoder/decoder_result.py?L120-150)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py@26f673d?L1-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/stacked_epoch_slices.py?L1-100)
- [type:commit (message:angular OR message:directional) repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis after:2024-01-01 count:30](/search?q=type:commit%20%28message:angular%20OR%20message:directional%29%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20after:2024-01-01%20count:30)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ file:.*\.py angular position decode](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20file:.%2A%5C.py%20angular%20position%20decode)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "DirectionalPseudo2DDecodersResult" OR "DirectionalDecodersContinuouslyDecodedResult"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22DirectionalPseudo2DDecodersResult%22%20OR%20%22DirectionalDecodersContinuouslyDecodedResult%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py@26f673d?L1-100](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/DecoderPredictionError.py?L1-100)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "_display_directional" OR "_display_decoded" OR "plot_directional" OR "angular_data"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22_display_directional%22%20OR%20%22_display_decoded%22%20OR%20%22plot_directional%22%20OR%20%22angular_data%22)

## Suggested Follow-ups

- Can you provide a code example demonstrating how to use the DirectionalPseudo2DDecodersResult class to decode position from neural activity data?
- Are there any example notebooks or tutorials that showcase the real-time visualization capabilities of the TimeSynchronizedPositionDecoderPlotter class with directional decoding?


---

# I remember a class or function that allows visualization of segments of 2D trajectories as "snapshots" or something similar for rendering them on a timeline, find this function

Excellent! I found the functions you're looking for. These are the key classes and functions that allow visualization of 2D trajectory segments as "snapshots" on a timeline:

## **Key Timeline Visualization Functions**

### **1. `_add_context_marginal_to_timeline` and `_add_context_decoded_epoch_marginals_to_timeline`**
- **File**: [`src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L662-L707)
- **Created**: 2025-09-10 (very recent!)
- **Purpose**: Add decoded position marginals and epoch segments to timeline visualizations

```python
@function_attributes(short_name=None, tags=['track', 'decoded-continuous'], 
                    creation_date='2025-09-10 11:56')
def _add_context_marginal_to_timeline(active_2d_plot, a_filter_epochs_decoded_result: SingleEpochDecodedResult, name='marginal_ctxt'):
    """Adds continuous decoded marginals to timeline"""
    
@function_attributes(short_name=None, tags=['track', 'multi-track', 'decoded-epochs'], 
                    creation_date='2025-09-10 11:55')
def _add_context_decoded_epoch_marginals_to_timeline(active_2d_plot, decoded_epochs_result: DecodedFilterEpochsResult, name: str = f'epochs_name[time_bin_size]'):
    """Adds decoded epoch segments as snapshots to timeline"""
```

### **2. Core Timeline Track Functions**
- **File**: [`src/pyphoplacecellanalysis/GUI/PyQtPlot/DockingWidgets/SpecificDockWidgetManipulatingMixin.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/DockingWidgets/SpecificDockWidgetManipulatingMixin.py)

#### **`add_docked_marginal_track`**
```python
def add_docked_marginal_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, 
                             xbin: Optional[NDArray]=None, a_variable_name: Optional[str]=None, 
                             a_dock_config: Optional[CustomDockDisplayConfig]=None, 
                             extended_dock_title_info: Optional[str]=None, 
                             posterior_heatmap_imshow_kwargs: Optional[Dict]=None):
    """adds a marginal (such as Long v. Short, or Long_LR v. Long_RL v. Short_LR v. Short_RL)"""
```

#### **`add_docked_decoded_posterior_slices_track`**
```python
def add_docked_decoded_posterior_slices_track(self, name: str, slices_time_window_centers: List[NDArray], 
                                             slices_posteriors: List[NDArray], 
                                             xbin: Optional[NDArray]=None, 
                                             measured_position_df: Optional[pd.DataFrame]=None, 
                                             a_variable_name: Optional[str]=None, 
                                             a_dock_config: Optional[CustomDockDisplayConfig]=None, 
                                             extended_dock_title_info: Optional[str]=None, 
                                             posterior_heatmap_imshow_kwargs: Optional[Dict]=None):
    """adds a decoded 1D posterior as slices/snapshots"""
```

### **3. AnimalTrajectoryPlottingMixin**
- **File**: [`src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/AnimalTrajectoryPlottingMixin.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/AnimalTrajectoryPlottingMixin.py)
- **Purpose**: Renders trajectory segments through space with temporal snapshots

Key properties:
- `curr_recent_trajectory`: Gets recent trajectory segments
- `recent_position_trajectory_max_seconds_ago`: Controls time window for trajectory snapshots
- `recent_position_trajectory_max_num_plotted_samples`: Limits number of trajectory points

### **4. Usage Example**

```python
from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_context_marginal_to_timeline, _add_context_decoded_epoch_marginals_to_timeline

# Add continuous marginals to timeline
_out = _add_context_marginal_to_timeline(active_2d_plot, 
                                        a_filter_epochs_decoded_result=all_context_filter_epochs_decoder_result, 
                                        name='global context')

# Add epoch slices/snapshots to timeline  
_out_pbe_tracks = _add_context_decoded_epoch_marginals_to_timeline(active_2d_plot=active_2d_plot, 
                                                                  decoded_epochs_result=pbe_decoder_result)
```

These functions allow you to visualize 2D trajectory segments as "snapshots" or slices on a timeline, where each segment represents a decoded position posterior at different time points. The system can handle both continuous decoding (marginals over time) and discrete epoch-based decoding (individual snapshots/slices).

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "snapshot" OR "timeline" OR "trajectory" OR "segment"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22snapshot%22%20OR%20%22timeline%22%20OR%20%22trajectory%22%20OR%20%22segment%22)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ trajectory segment timeline visualization](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20trajectory%20segment%20timeline%20visualization)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "2D" AND ("trajectory" OR "path" OR "position") AND ("render" OR "plot" OR "visualiz")](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%222D%22%20AND%20%28%22trajectory%22%20OR%20%22path%22%20OR%20%22position%22%29%20AND%20%28%22render%22%20OR%20%22plot%22%20OR%20%22visualiz%22%29)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "_add_context_marginal_to_timeline" OR "_add_context_decoded_epoch_marginals_to_timeline"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22_add_context_marginal_to_timeline%22%20OR%20%22_add_context_decoded_epoch_marginals_to_timeline%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@26f673d?L660-720](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L660-720)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "AnimalTrajectoryPlottingMixin" OR "curr_recent_trajectory"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22AnimalTrajectoryPlottingMixin%22%20OR%20%22curr_recent_trajectory%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/AnimalTrajectoryPlottingMixin.py@26f673d?L40-150](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/AnimalTrajectoryPlottingMixin.py?L40-150)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ "add_docked_marginal_track" OR "add_docked_decoded_posterior_slices_track"](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20%22add_docked_marginal_track%22%20OR%20%22add_docked_decoded_posterior_slices_track%22)

## Suggested Follow-ups

- Could you provide a minimal working example demonstrating how to use `_add_context_decoded_epoch_marginals_to_timeline` to display trajectory snapshots for a single epoch?
- How can I customize the appearance of the trajectory snapshots, such as color, opacity, and size, when using `_add_context_decoded_epoch_marginals_to_timeline`?

