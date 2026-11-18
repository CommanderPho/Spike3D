---
title: Find colored curve rendering and colormaps
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/cc81f90d-3f65-4412-b149-cc1435b1273a
url: https://phohale.sourcegraph.app/deepsearch/330
created: 2026-01-21T07:50:36Z
updated: 2026-07-22T18:07:31Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis find all instances where I've rendered colored lines/curves in my codebase, especially those used to represent trajectories or paths through space. Also anything related to building custom colormaps would be helpful.

Perfect! Now I have a comprehensive understanding of the colored line/trajectory rendering in your codebase. Let me create a summary with the key findings:

Based on my search through your codebase, I found several key instances where you render colored lines/curves for trajectories and paths, along with custom colormap implementations. Here are the main findings:

## **Colored Trajectory Lines/Curves**

### 1. **Gradient Line Helper for Time-Based Trajectories** 
The main implementation is in [`_helper_add_gradient_line`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1485-L1521):

```python
def _helper_add_gradient_line(cls, ax, t, x, y, add_markers=False, time_cmap='viridis', **LineCollection_kwargs):
    """ Adds a gradient line representing a timeseries of (x, y) positions. """
    norm = plt.Normalize(t.min(), t.max())
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    if isinstance(time_cmap, str):
        time_cmap = plt.get_cmap(time_cmap)
    
    lc = LineCollection(segments, cmap=time_cmap, norm=norm, **LineCollection_kwargs)
    lc.set_array(t)
    lc.set_linewidth(2)
    lc.set_alpha(0.85)
    line = ax.add_collection(lc)
```

This is used in [`decoder_plotting_mixins.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1701-L1709) to plot measured position with a green-to-black gradient:

```python
colors = [(0, 0.6, 0), (0, 0, 0)]  # green to black
time_cmap = LinearSegmentedColormap.from_list("GreenToBlack", colors, N=25)
a_meas_pos_line, _meas_pos_out_markers = cls._helper_add_gradient_line(
    an_ax, t=a_measured_time_bin_centers, **pos_kwargs, 
    add_markers=add_markers, time_cmap=time_cmap, zorder=0
)
```

### 2. **Lap Trajectory Plotting**
In [`laps.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py?L473-L507), trajectories through space are rendered with time-gradient coloring:

```python
# Create time-gradient colored trajectory
curr_lap_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
norm = plt.Normalize(curr_lap_timeseries.min(), curr_lap_timeseries.max())
points = np.array([laps_position_traces[curr_lap_id][0,:], laps_position_traces[curr_lap_id][1,:]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(curr_lap_timeseries)
lc.set_linewidth(2)
lc.set_alpha(0.85)
```

### 3. **Track Shape Drawing with LineCollection**
In [`track_shape_drawing.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py?L1075-L1087), vertical/horizontal lines for track boundaries:

```python
long_track_line_collection: matplotlib.collections.LineCollection = ax.vlines(
    long_notable_x_platform_positions, label='long_track_x_pos_lines', 
    colors=long_kwargs.get('edgecolor', '#0000FFAA'), 
    linewidths=long_kwargs.get('linewidth', 1.0), 
    linestyles='dashed'
)
```

### 4. **Animal Trajectory Plotting (PyQtGraph)**
In [`AnimalTrajectoryPlottingMixin.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/AnimalTrajectoryPlottingMixin.py?L98-L106):

```python
self.ui.trajectory_curve = pg.PlotDataItem(
    pen=self.params.recent_position_trajectory_path_pen, 
    shadowPen=self.params.recent_position_trajectory_path_shadow_pen,
    symbol='o', 
    symbolBrush=self.params.recent_position_trajectory_symbol_brush, 
    symbolPen=self.params.recent_position_trajectory_symbol_pen,
    antialias=True, 
    name='recent trajectory'
)
```

## **Custom Colormap Building**

### 1. **Custom Colormaps with Transparency/Low-Value Cutoffs**
In [`LongShortDisplayConfig.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/LongShortDisplayConfig.py?L758-L835):

**Orange colormap with dropped low values:**
```python
def get_custom_orange_with_low_values_dropped_cmap(cls):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f"dropping_low_values_oranges_colormap", [
        [0.87306,0.62718,0.34353,0.     ],  # transparent at low values
        [0.87306,0.62718,0.34353,1.     ],
        [0.80485,0.48515,0.24302,1.     ],
        [0.76402,0.36984,0.181  ,1.     ],
        [0.72319,0.28559,0.13887,1.     ],
        [0.65098,0.19608,0.10588,1.     ]
    ])
```

**Greyscale with adjustable transparency:**
```python
def get_custom_greyscale_with_low_values_dropped_cmap(cls, low_value_cutoff=0.05, full_opacity_threshold=0.4, grey_value=0.1, invert_for_black_bg=False):
    final_color = [1.0, 1.0, 1.0, 1.0] if invert_for_black_bg else [0.0, 0.0, 0.0, 1.0]
    colors = [
        (0.0, [grey_value, grey_value, grey_value, 0.0]),
        (low_value_cutoff, [grey_value, grey_value, grey_value, 0.0]),
        (full_opacity_threshold, [grey_value, grey_value, grey_value, 1.0]),
        (1.0, final_color)
    ]
    return LinearSegmentedColormap.from_list(colormap_name, colors)
```

### 2. **ColormapHelpers.create_transparent_colormap**
Referenced in multiple files ([`TrialByTrialActivityWindow.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/TrialByTrialActivityWindow.py?L157), [`PendingNotebookCode.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L3617)):

```python
# Create transparent colormap from color name or literal
active_decoder_cmap_dict = {
    k: ColormapHelpers.create_transparent_colormap(
        color_literal_name=v, 
        lower_bound_alpha=0.1, 
        should_return_LinearSegmentedColormap=True
    ) 
    for k, v in all_decoder_colors_dict.items()
}
```

### 3. **PyQtGraph Colormap Conversion**
In [`pyqtgraph_helpers.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/Extensions/pyqtgraph_helpers.py?L108-L113):

```python
cmap = pg.colormap.get('Oranges','matplotlib')
root_render_widget.setColorMap(cmap)
```

### 4. **Vedo Custom Spike Colormaps**
In [`Vedo3DStatic.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/GUI/Vedo/Vedo3DStatic.py?L50-L68):

```python
def _build_spikes_colormap(cls, spike_color_ids):
    try:
        import colorcet
        mycmap = colorcet.glasbey_light
        alphas = np.linspace(0.8, 0.2, num=len(mycmap))
    except:
        mycmap = ["darkblue", "magenta", (1, 1, 0)]
        alphas = [0.8, 0.6, 0.2]
    return mycmap, alphas, spike_color_ids
```

## **Key Files to Explore**

1. **[`decoder_plotting_mixins.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py)** - Main gradient line/trajectory rendering
2. **[`laps.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py)** - Lap trajectory visualization with time gradients
3. **[`LongShortDisplayConfig.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/LongShortDisplayConfig.py)** - Custom colormap definitions
4. **[`track_shape_drawing.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py)** - Track boundary line rendering
5. **[`AnimalTrajectoryPlottingMixin.py`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Pho2D/PyQtPlots/TimeSynchronizedPlotters/Mixins/AnimalTrajectoryPlottingMixin.py)** - Real-time trajectory plotting

The core pattern across all implementations uses `matplotlib.collections.LineCollection` with segments and colormaps to create gradient-colored trajectories representing movement through space over time.

## Sources

- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis colored line curve trajectory path render](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20colored%20line%20curve%20trajectory%20path%20render)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis colormap custom color map build create](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20colormap%20custom%20color%20map%20build%20create)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis LineCollection](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20LineCollection)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis set_color OR set_colors OR set_array](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20set_color%20OR%20set_colors%20OR%20set_array)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis ListedColormap OR LinearSegmentedColormap](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20ListedColormap%20OR%20LinearSegmentedColormap)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis plot_trajectory OR trajectory_plot](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20plot_trajectory%20OR%20trajectory_plot)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis segments cmap norm](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20segments%20cmap%20norm)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis create_transparent_colormap](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20create_transparent_colormap)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis gradient line trajectory](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20gradient%20line%20trajectory)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis matplotlib colormap alpha transparent custom](/search?q=patterntype:nls%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20matplotlib%20colormap%20alpha%20transparent%20custom)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis LinearSegmentedColormap.from_list](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20LinearSegmentedColormap.from_list)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis class ColormapHelpers](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20class%20ColormapHelpers)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py@26f673d?L1481-1530](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py?L1481-1530)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/Configs/LongShortDisplayConfig.py@26f673d?L757-835](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Model/Configs/LongShortDisplayConfig.py?L757-835)
- [repo:github.com/CommanderPho/pyPhoPlaceCellAnalysis def create_transparent_colormap](/search?q=patterntype:keyword%20repo:github.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis%20def%20create_transparent_colormap)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py@26f673d?L473-530](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/PhoPositionalData/plotting/laps.py?L473-530)

## Suggested Follow-ups

- Are there any performance considerations when using `LineCollection` for long trajectories, and are there alternative approaches for optimizing trajectory rendering?
- Can you provide a more detailed breakdown of the colorcet colormap integration in `Vedo3DStatic.py`, specifically how the colors are mapped to spike data and how I can customize this mapping?

