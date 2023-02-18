---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: spike3d-poetry
    language: python
    name: spike3d-poetry
---

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
# Purpose
This notebook serves to contain the final, mostly user-level documentation for Spike3D
<!-- #endregion -->

<!-- #region tags=[] -->
# `SpikeRaster2D`, `SpikeRaster3D`, and `Spike3DRasterWindowWidget`
<!-- #endregion -->

```python
# Create a new `SpikeRaster2D` instance using `_display_spike_raster_pyqtplot_2D` and capture its outputs:
active_2d_plot, active_3d_plot, spike_raster_window = curr_active_pipeline.plot._display_spike_rasters_pyqtplot_2D.values()
```

<!-- #region tags=[] -->
### Getting the existing `Spike3DRasterWindowWidget`
<!-- #endregion -->

```python
from pyphocorehelpers.gui.Qt.TopLevelWindowHelper import TopLevelWindowHelper
import pyphoplacecellanalysis.External.pyqtgraph as pg # Used to get the app for TopLevelWindowHelper.top_level_windows
## For searching with `TopLevelWindowHelper.all_widgets(...)`:
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike3DRaster import Spike3DRaster
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

found_spike_raster_windows = TopLevelWindowHelper.all_widgets(pg.mkQApp(), searchType=Spike3DRasterWindowWidget)
assert len(found_spike_raster_windows) == 1, f"found {len(found_spike_raster_windows)} Spike3DRasterWindowWidget windows using TopLevelWindowHelper.all_widgets(...) but require exactly one."
spike_raster_window = found_spike_raster_windows[0]
# Extras:
active_2d_plot = spike_raster_window.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
active_3d_plot = spike_raster_window.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
## Rendered Time Curves Documentation Guide

#### `PyQtGraphSpecificTimeCurvesMixin(TimeCurvesViewMixin)`: mostly overriden for Spike2DRaster, but defines main plotting functions for Spike3DRaster
pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.RenderTimeCurvesMixin.PyQtGraphSpecificTimeCurvesMixin

<!-- #endregion -->

```python
add_3D_time_curves
```

```python
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import GeneralRenderTimeCurves
from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable
```

### For `Spike2DRaster`

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Requires

self.params.time_curves_no_update

## Single Datasource for time-curves:
self.params.time_curves_datasource


<!-- #endregion -->

#### Provides

`self.ui.main_time_curves_view_widget`
`self.ui.main_time_curves_view_legend`




#### Functions
clear_all_3D_time_curves(self)
update_3D_time_curves(self)

_build_or_update_time_curves_legend(self)
---
##### `_build_or_update_time_curves_plot`: uses or builds a new self.ui.main_time_curves_view_widget, which the item is added to
_build_or_update_time_curves_plot(self, plot_name, points, **kwargs)

---
update_3D_time_curves_baseline_grid_mesh
remove_3D_time_curves_baseline_grid_mesh



#### TimeCurvesViewMixin/PyQtGraphSpecificTimeCurvesMixin specific overrides for 2D:
""" 
As soon as the first 2D Time Curve plot is needed, it creates:
    self.ui.main_time_curves_view_widget - PlotItem by calling add_separate_render_time_curves_plot_item(...)

main_time_curves_view_widget creates new PlotDataItems by calling self.ui.main_time_curves_view_widget.plot(...)
    This .plot(...) command can take either: 
        .plot(x=x, y=y)
        .plot(ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]

"""

```python

```

<!-- #region tags=["procedure"] -->
### Procedure: Adding new Curves:
1. Copy pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves.PositionRenderTimeCurves into a new structure, changing as needed to display your desired variables
2. Add your new curve class to the import list at the top of `pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable` 
3. Use QtDesigner to add your menu in `GUI/Qt/Menus/LocalMenus_AddRenderable/LocalMenus_AddRenderable.ui` with an appropriate name.
	1. The objectName must follow the convention: `actionAddTimeCurves_Position`  -> e.g. `actionAddTimeCurves_Velocity`
4. Save and compile the .ui file (In VSCode: Right click > Compile .ui file)
5. Inside `LocalMenus_AddRenderable.build_renderable_menu(...)` add the appropriate entry to the `submenu_addTimeCurves` and `submenu_addTimeCurvesCallbacks` arrays.
	1. `lambda evt=None: VelocityRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),`
<!-- #endregion -->

<!-- #region tags=["TODO"] -->
##### TODO: Time-curve adding improvements
Enable users to 'register' new curves which are then added to the menu and the plot
<!-- #endregion -->

<!-- #region tags=[] -->
## Screenshots
<!-- #endregion -->

![[WithPBE_Epochs.png|500]]

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
# `matplotlib_view_widget`
<!-- #endregion -->

### Dynamic Matplotlib Plots in Spike2DRaster

<!-- #region -->
`self.ui.matplotlib_view_widget`

In `Spike2DRaster`
```python
self.ui.dynamic_docked_widget_container = NestedDockAreaWidget()
```
Helper Functions:
```python
# matplotlib render subplot __________________________________________________________________________________________ #
    def add_new_matplotlib_render_plot_widget(self, row=1, col=0, name='matplotlib_view_widget'):
        """ creates a new MatplotlibTimeSynchronizedWidget, a container widget that holds a matplotlib figure, and adds it as a row to the main layout """

    def remove_matplotlib_render_plot_widget(self):
        """ removes the subplot - does not work yet """

    def sync_matplotlib_render_plot_widget(self):
        """ Perform Initial (one-time) update from source -> controlled: """

    def clear_all_matplotlib_plots(self):
        """ required by the menu function """

```
<!-- #endregion -->

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
# ◽📣 Rectangle Epoch Documentation Guide
<!-- #endregion -->

<!-- #region -->
IntervalsDatasource

C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\Datasources\IntervalDatasource.py

C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\EpochRenderingMixin.py
C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\RenderTimeEpoch3DMeshesMixin.py

#### `Render2DEventRectanglesHelper`:
GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/Render2DEventRectanglesHelper.py

#### `Specific2DRenderTimeEpochs`:
GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/Specific2DRenderTimeEpochs.py


#### `Spike2DRaster`
_perform_add_render_item
_perform_remove_render_item
add_laps_intervals/remove_laps_intervals
add_PBEs_intervals/remove_PBEs_intervals


<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## Screenshots
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
### 3D Interval Rects
#rectangles #IntervalRectsItem #interval #PBEs #3d #spike3d 


Here you can see many short intervals rendered as cyan rectangles on the floor of the 3D Raster
<!-- #endregion -->

<!-- #region scene__Default Scene=true pycharm={"is_executing": false, "name": "#%%\n"} tags=["ActiveScene", "gui", "launch", "main_run"] -->
`active_3d_plot.add_rendered_intervals(new_ripples_intervals_datasource, name='new_ripples')`
<!-- #endregion -->

<!-- #region scene__Default Scene=true pycharm={"is_executing": false, "name": "#%%\n"} tags=["ActiveScene", "gui", "launch", "main_run"] -->
![python_JwdIMVHpEQ.png](attachment:52498aab-31a8-4a0b-8add-0728809de9ab.png)
![image.png](attachment:dabc70cf-76b1-45b6-b7a0-a3bf785e5391.png)
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## ◽📣 ✅ Testing 2D Rectangle Epochs on Raster Plot
<!-- #endregion -->

```python
laps_interval_datasource = Specific2DRenderTimeEpochsHelper.build_Laps_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=max_series_top, series_height=1.0) # series_vertical_offset=42.0
new_PBEs_interval_datasource = Specific2DRenderTimeEpochsHelper.build_PBEs_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=(max_series_top+1.0), series_height=3.0) # new_PBEs_interval_datasource

## General Adding:
active_2d_plot.add_rendered_intervals(new_PBEs_interval_datasource, name='PBEs', child_plots=[background_static_scroll_plot_widget, main_plot_widget], debug_print=False)
active_2d_plot.add_rendered_intervals(laps_interval_datasource, name='Laps', child_plots=[background_static_scroll_plot_widget, main_plot_widget], debug_print=False)
```

```python
active_2d_plot.add_laps_intervals(sess)
```

```python
active_2d_plot.remove_laps_intervals()
```

```python
# active_2d_plot.add_PBEs_intervals(sess)
```

```python
active_2d_plot.interval_rendering_plots
```

```python
active_2d_plot.clear_all_rendered_intervals()
```

```python
interval_info = active_2d_plot.list_all_rendered_intervals()
interval_info
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## 📣 Programmatically adding several epoch rectangles by calling the addRenderable context menu functions all at once for SpikeRaster2D
<!-- #endregion -->

```python
add_renderables_menu = active_2d_plot.ui.menus.custom_context_menus.add_renderables[0].programmatic_actions_dict
menu_commands = ['AddTimeIntervals.PBEs', 'AddTimeIntervals.Ripples', 'AddTimeIntervals.Replays', 'AddTimeIntervals.Laps', 'AddTimeIntervals.Session.Epochs']
for a_command in menu_commands:
    add_renderables_menu[a_command].trigger()    
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## ◽📣 Updating Epochs visual appearance
<!-- #endregion -->

```python
interval_info = active_2d_plot.list_all_rendered_intervals()
interval_info
```

```python
active_2d_plot.clear_all_rendered_intervals()
```

```python
active_2d_plot.interval_rendering_plots
```

```python
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, Ripples_2DRenderTimeEpochs

# Need to deal with pg.mkPen(a_pen_color) and pg.mkBrush
def build_custom_epochs_dataframe_formatter(cls, **kwargs):
    def _add_interval_dataframe_visualization_columns_general_epoch(active_df):
        """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
        """
        num_intervals = np.shape(active_df)[0]
        ## parameters:
        y_location = 0.0
        height = 20.5
        pen_color = pg.mkColor('w')
        pen_color.setAlphaF(0.8)

        brush_color = pg.mkColor('grey')
        brush_color.setAlphaF(0.5)

        ## Update the dataframe's visualization columns:
        active_df = cls._update_df_visualization_columns(active_df, y_location=y_location, height=height, pen_color=pen_color, brush_color=brush_color, **kwargs)
        return active_df
    return _add_interval_dataframe_visualization_columns_general_epoch

interval_datasource = Ripples_2DRenderTimeEpochs.build_render_time_epochs_datasource(sess.laps.as_epoch_obj(), epochs_dataframe_formatter=build_custom_epochs_dataframe_formatter) # **({'series_vertical_offset': 42.0, 'series_height': 1.0} | kwargs)
spike_raster_window.spike_raster_plt_2d.add_rendered_intervals(interval_datasource, name='CustomRipples', debug_print=False) # removes the rendered intervals
```

<!-- #region scene__Default Scene=true pycharm={"is_executing": false, "name": "#%%\n"} tags=["ActiveScene", "gui", "launch", "main_run"] -->
### Concise Update:
<!-- #endregion -->

```python scene__Default Scene=true pycharm={"is_executing": false, "name": "#%%\n"} tags=["ActiveScene", "gui", "launch", "main_run"]
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, Ripples_2DRenderTimeEpochs, inline_mkColor
## Inline Concise: Position Replays, PBEs, and Ripples all below the scatter:
# active_2d_plot.interval_datasources.Replays.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-10.0, height=7.5, pen_color=inline_mkColor('orange', 0.8), brush_color=inline_mkColor('orange', 0.5), **kwargs)) ## Fully inline
# active_2d_plot.interval_datasources.PBEs.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-2.0, height=1.5, pen_color=inline_mkColor('pink', 0.8), brush_color=inline_mkColor('pink', 0.5), **kwargs)) ## Fully inline
# active_2d_plot.interval_datasources.Ripples.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5), **kwargs)) ## Fully inline
# active_2d_plot.interval_datasources.SessionEpochs .update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5), **kwargs)) ## Fully inline
epochs_update_dict = {
    'Replays':dict(y_location=-10.0, height=7.5, pen_color=inline_mkColor('orange', 0.8), brush_color=inline_mkColor('orange', 0.5)),
    'PBEs':dict(y_location=-2.0, height=1.5, pen_color=inline_mkColor('pink', 0.8), brush_color=inline_mkColor('pink', 0.5)),
    'Ripples':dict(y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5)),
    'SessionEpochs ':dict(y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5)),
}
active_2d_plot.update_rendered_intervals_visualization_properties(epochs_update_dict)

```

<!-- #region scene__Default Scene=true jp-MarkdownHeadingCollapsed=true pycharm={"is_executing": false, "name": "#%%\n"} tags=["ActiveScene", "gui", "launch", "main_run"] -->
### Build Stacked Layout:
<!-- #endregion -->

```python scene__Default Scene=true pycharm={"is_executing": false, "name": "#%%\n"} tags=["ActiveScene", "gui", "launch", "main_run"]
rendered_interval_keys = list(interval_info.keys())
desired_interval_height_ratios = [2.0, 2.0, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval
required_vertical_offsets, required_interval_heights = EpochRenderingMixin.build_stacked_epoch_layout(desired_interval_height_ratios, epoch_render_stack_height=20.0, interval_stack_location='below')
stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(rendered_interval_keys, required_vertical_offsets, required_interval_heights)} # Build a stacked_epoch_layout_dict to update the display
active_2d_plot.update_rendered_intervals_visualization_properties(stacked_epoch_layout_dict)
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## ◽📣 Get list of existing interval rect datasources:
blah blah
<!-- #endregion -->

```python
# Plot items:
active_2d_plot.interval_rendering_plots
```

```python
# active_2d_plot.interval_datasources.new_ripples
interval_info = active_2d_plot.list_all_rendered_intervals()
interval_info
```

```python
active_2d_plot.interval_datasources # RenderPlotsData
# datasource_to_update
```

```python
active_2d_plot.interval_datasources.PBEs # IntervalsDatasource
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## ◽📣 Update existing interval rects:
Write a function that takes your existing datasource dataframe and updates its columns.
<!-- #endregion -->

<!-- #region tags=[] -->
### Before Update:
![python_YwFQ3gs3K2.png](attachment:d13ff2db-bf10-457f-8184-9cf4f822eb38.png)
<!-- #endregion -->

```python
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, Ripples_2DRenderTimeEpochs
# series_vertical_offset, series_height, pen, brush

def _updated_custom_interval_dataframe_visualization_columns_general_epoch(active_df, **kwargs):
    """ Adds the remaining _required_interval_visualization_columns specifically for PBEs
    """
    num_intervals = np.shape(active_df)[0]
    ## parameters:
    y_location = 0.0
    height = 30.5
    pen_color = pg.mkColor('grey')
    pen_color.setAlphaF(0.8)

    brush_color = pg.mkColor('grey')
    brush_color.setAlphaF(0.5)

    ## Update the dataframe's visualization columns:
    active_df = General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=y_location, height=height, pen_color=pen_color, brush_color=brush_color, **kwargs)
    return active_df

# get the existing dataframe to be updated:
# datasource_to_update = active_2d_plot.interval_datasources.Ripples
datasource_to_update = active_2d_plot.interval_datasources.new_ripples
# datasource_to_update = active_2d_plot.interval_datasources.CustomRipples
datasource_to_update.update_visualization_properties(_updated_custom_interval_dataframe_visualization_columns_general_epoch)
```

```python
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, inline_mkColor
## Inline Concise: Position Replays, PBEs, and Ripples all below the scatter:
active_2d_plot.interval_datasources.Replays.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-10.0, height=7.5, pen_color=inline_mkColor('orange', 0.8), brush_color=inline_mkColor('orange', 0.5), **kwargs)) ## Fully inline
active_2d_plot.interval_datasources.PBEs.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-2.0, height=1.5, pen_color=inline_mkColor('pink', 0.8), brush_color=inline_mkColor('pink', 0.5), **kwargs)) ## Fully inline
active_2d_plot.interval_datasources.Ripples.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5), **kwargs)) ## Fully inline
```

### Post Update:
![python_LKGNtQCkQH.png](attachment:99906b90-2fdd-42ec-8536-ec0e52b73c68.png)

```python
datasource_to_update.custom_datasource_name
```

```python
datasource_to_update.df
```

```python
spike_raster_window.spike_raster_plt_2d.add_rendered_intervals(datasource_to_update, name='CustomRipples', debug_print=True) 
```

```python
# ## Global main plot (doesn't work)
# fig_global, main_ax = plt.subplots()
# # fig_global, (main_ax) = plt.subplots(1, 1)

# Plot a line in the first axes.
main_ax.plot(ripple_predictions_df.t.to_numpy(), ripple_predictions_df.v.to_numpy(), "-o")
main_ax.set_title(f'global predicted ripple probability: shank {shank_id}')

# # Create a view! Turn axes 2 into a view of axes 1.
# view(ax2, ax1)

# # Modify the second axes data limits so we get a slightly zoomed out view
# ax2.set_xlim(-5, 15)
# ax2.set_ylim(-5, 15)

# fig_global.show()
```

```python
plots.fig.show()
```

```python
# laps_position_times_list = [np.squeeze(lap_pos_df[['t']].to_numpy()) for lap_pos_df in lap_specific_position_dfs]
# laps_position_traces_list = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]

# epochs = sess.laps.to_dataframe()
# epoch_slices = epochs[['start', 'stop']].to_numpy()
# epoch_description_list = [f'lap {epoch_tuple.lap_id} (maze: {epoch_tuple.maze_id}, direction: {epoch_tuple.lap_dir})' for epoch_tuple in epochs[['lap_id','maze_id','lap_dir']].itertuples()]
# print(f'epoch_description_list: {epoch_description_list}')


from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import stacked_epoch_slices_view

stacked_epoch_slices_view_laps_containers = stacked_epoch_slices_view(epoch_slices, laps_position_times_list, laps_position_traces_list, name=f'stacked_epoch_slices_view_new_ripples: shank {shank_id}')
params, plots_data, plots, ui = stacked_epoch_slices_view_laps_containers
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
## ◽📣 Removing/Clearing existing interval rects:
<!-- #endregion -->

### Selectively Removing:

```python
active_2d_plot.remove_rendered_intervals(name='PBEs', child_plots_removal_list=[main_plot_widget]) # Tests removing a single series from a single plot (main_plot_widget)
active_2d_plot.remove_rendered_intervals(name='PBEs') # Tests removing a single series ('PBEs') from all plots it's on
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
# 3D (PyVista/Vedo/etc)-based plots:
<!-- #endregion -->

```python pycharm={"is_executing": false, "name": "#%%\n"}
curr_active_pipeline.display('_display_3d_interactive_spike_and_behavior_browser', active_config_name) # this works now!
```

```python pycharm={"is_executing": false, "name": "#%%\n"}
display_dict = curr_active_pipeline.display('_display_3d_interactive_custom_data_explorer', active_config_name) # does not work, missing color info?
iplapsDataExplorer = display_dict['iplapsDataExplorer']
# plotter is available at
p = display_dict['plotter']
iplapsDataExplorer
```

```python
# curr_kdiba_pipeline.display(DefaultDisplayFunctions._display_3d_interactive_custom_data_explorer, 'maze1') # works!
curr_active_pipeline.display('_display_3d_interactive_tuning_curves_plotter', 'maze1_PYR') # works!
```

```python

```

<!-- #region -->
### Adjusting Spike Emphasis:
#### Usage Examples:
```python
from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState

## Example 1: De-emphasize spikes excluded from the placefield calculations:
is_spike_included_in_pf = np.isin(spike_raster_window.spike_raster_plt_2d.spikes_df.index, active_pf_2D.filtered_spikes_df.index)
spike_raster_window.spike_raster_plt_2d.update_spike_emphasis(np.logical_not(is_spike_included_in_pf), SpikeEmphasisState.Deemphasized)

## Example 2: De-emphasize spikes that don't have their 'aclu' from a given set of indicies:
is_spike_included = spike_raster_window.spike_raster_plt_2d.spikes_df.aclu.to_numpy() == 2
spike_raster_window.spike_raster_plt_2d.update_spike_emphasis(np.logical_not(is_spike_included), SpikeEmphasisState.Deemphasized)

## Example 3: De-emphasize all spikes 
active_2d_plot.update_spike_emphasis(new_emphasis_state=SpikeEmphasisState.Deemphasized)

## Example 4: Hide all spikes entirely
active_2d_plot.update_spike_emphasis(new_emphasis_state=SpikeEmphasisState.Hidden)
```

#### Notes
Looks like there is very advanced emphasis functionality that I haven't explored. See Code example below:
```python

# SpikeEmphasisState
state_alpha = {SpikeEmphasisState.Hidden: 0.01,
			   SpikeEmphasisState.Deemphasized: 0.1,
			   SpikeEmphasisState.Default: 0.5,
			   SpikeEmphasisState.Emphasized: 1.0,
}

# state_color_adjust_fcns: functions that take the base color and call build_adjusted_color to get the adjusted color for each state
state_color_adjust_fcns = {SpikeEmphasisState.Hidden: lambda x: build_adjusted_color(x),
			   SpikeEmphasisState.Deemphasized: lambda x: build_adjusted_color(x, saturation_scale=0.35, value_scale=0.8),
			   SpikeEmphasisState.Default: lambda x: build_adjusted_color(x),
			   SpikeEmphasisState.Emphasized: lambda x: build_adjusted_color(x, value_scale=1.25),
}

```
<!-- #endregion -->

<!-- #region -->
### Assigning Cell Colors
Working calls:
```python

## Set the colors of the raster window from the ipcDataExplorer window:
spike_raster_window.update_neurons_color_data(updated_neuron_render_configs=ipcDataExplorer.active_neuron_render_configs_map)


```

```python

""" Cell Coloring functions:
"""
def _setup_neurons_color_data(self, neuron_colors_list=None, coloring_mode='color_by_index_order'):
	""" 
	neuron_colors_list: a list of neuron colors
		if None provided will call DataSeriesColorHelpers._build_cell_color_map(...) to build them.
	
	Requires:
		self.fragile_linear_neuron_IDXs
		self.n_cells
	
	Sets:
		self.params.neuron_qcolors
		self.params.neuron_qcolors_map
		self.params.neuron_colors: ndarray of shape (4, self.n_cells)
		self.params.neuron_colors_hex
		

	Known Calls: Seemingly only called from:
		SpikesRenderingBaseMixin.helper_setup_neuron_colors_and_order(...)
	"""

def update_neurons_color_data(self, updated_neuron_render_configs):
        """updates the colors for each neuron/cell given the updated_neuron_render_configs map
        updated_neuron_render_configs: {2: SingleNeuronPlottingExtended(color='#843c39', extended_values_dictionary={}, isVisible=False, name='2', spikesVisible=False),
		 3: SingleNeuronPlottingExtended(color='#924744', extended_values_dictionary={}, isVisible=False, name='3', spikesVisible=False),
		 4: SingleNeuronPlottingExtended(color='#9f5350', extended_values_dictionary={}, isVisible=False, name='4', spikesVisible=False),
		 ... ,
		 109: SingleNeuronPlottingExtended(color='#f0aee7', extended_values_dictionary={}, isVisible=False, name='109', spikesVisible=False)}
 
```
<!-- #endregion -->

```python

```

```python

```

<!-- #region tags=[] -->
# 🏻‍💻 DEVELOPER SECTION
<!-- #endregion -->

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
## TODO/PENDING
<!-- #endregion -->

#### Set the tooltip for each individual rect so that when you hover a rect it shows relevant information (that intervals series name, its (start, end) times, duration, index, etc
search code in cell below to find where it's set generally. Try `CustomIntervalRectsItem`
![image.png](attachment:c06368e0-a801-4e50-8c41-dbe9eb6d50eb.png)

```python
# Build the rendered interval item:
new_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource)
new_interval_rects_item.setToolTip(name) # The tooltip is set generically here to 'PBEs', 'Replays' or whatever the dataseries name is
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## 👨🏻‍💻📚 Computation Functions Documentation Guide
<!-- #endregion -->

```python
curr_active_pipeline.global_computation_results
```

```python
curr_active_pipeline.computation_results
```

### Registering a new computation function

```python
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import MultiContextComputationFunctions

curr_active_pipeline.register_computation(computation_function=MultiContextComputationFunctions._perform_jonathan_replay_firing_rate_analyses, is_global=True, registered_name='_perform_jonathan_replay_firing_rate_analyses')
```

### Performing a specific computation:

```python
curr_active_pipeline.perform_specific_computation(computation_functions_name_whitelist=['_perform_jonathan_replay_firing_rate_analyses'], fail_on_exception=True, debug_print=True) # , progress_logger_callback=print
```

```python
curr_active_pipeline.save_pipeline()
```

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true -->
## Computation Classes Documentation
<!-- #endregion -->

<!-- #region tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## Time-Dependent Placefields Documentation:

### Resetting State:
reset(self): """ used to reset the calculations to an initial value. """
    setup_time_varying(self): """ Initialize for the 0th timestamp """

### Making Snapshots:
snapshot(self): """ takes a snapshot of the current values at this time."""    
        
### Restore Snapshots:
restore_from_snapshot(self, snapshot_t)
    apply_snapshot_data(self, snapshot_t, snapshot_data)
  
<!-- #endregion -->

```python
# Reset the rebuild_fragile_linear_neuron_IDXs:
self._filtered_spikes_df, _reverse_cellID_index_map = self._filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
self.fragile_linear_neuron_IDXs = np.unique(self._filtered_spikes_df.fragile_linear_neuron_IDX) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])
self.n_fragile_linear_neuron_IDXs = len(self.fragile_linear_neuron_IDXs)
self._included_thresh_neurons_indx = np.arange(self.n_fragile_linear_neuron_IDXs)
self._peak_frate_filter_function = lambda list_: [list_[_] for _ in self._included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria
# ...
self.setup_time_varying()
```

```python
## reset(...)
self.curr_spikes_maps_matrix = np.zeros((self.n_fragile_linear_neuron_IDXs, *dims_coord_tuple), dtype=int) # create an initially zero occupancy map
self.curr_smoothed_spikes_maps_matrix = None
self.curr_num_pos_samples_occupancy_map = np.zeros(dims_coord_tuple, dtype=int) # create an initially zero occupancy map
self.curr_num_pos_samples_smoothed_occupancy_map = None
self.last_t = 0.0
self.curr_seconds_occupancy = np.zeros(dims_coord_tuple, dtype=float)
self.curr_normalized_occupancy = self.curr_seconds_occupancy.copy()
self.curr_occupancy_weighted_tuning_maps_matrix = np.zeros((self.n_fragile_linear_neuron_IDXs, *dims_coord_tuple), dtype=float) # will have units of # spikes/sec
self.historical_snapshots = OrderedDict({})
```

<!-- #region jupyter={"outputs_hidden": false} jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true -->
## Data Structure Documentation Generation
The functions below generate documentation in .md and .html format from passed data structures.
<!-- #endregion -->


```python jupyter={"outputs_hidden": false}
print_keys_if_possible('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'], custom_item_formatter=_rich_text_format_curr_value)
```


```python jupyter={"outputs_hidden": false}
from ansi2html import Ansi2HTMLConverter # used by DocumentationFilePrinter to build html document from ansi-color coded version
from pyphocorehelpers.print_helpers import DocumentationFilePrinter

doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])
```


```python jupyter={"outputs_hidden": false} tags=[]
from ansi2html import Ansi2HTMLConverter # used by DocumentationFilePrinter to build html document from ansi-color coded version
from pyphocorehelpers.print_helpers import DocumentationFilePrinter

doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='InteractivePlaceCellConfig')
doc_printer.save_documentation('InteractivePlaceCellConfig', curr_active_pipeline.active_configs['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map', 'pf_listed_colormap'])
# doc_printer.reveal_output_files_in_system_file_manager()
```


```python jupyter={"outputs_hidden": false} tags=[]
doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='NeuropyPipeline')
doc_printer.save_documentation('NeuropyPipeline', curr_active_pipeline, non_expanded_item_keys=['_reverse_cellID_index_map', 'pf_listed_colormap', 'computation_results', 'active_configs', 'logger']) # 'Logger'
```


```python jupyter={"outputs_hidden": false} tags=[]
doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='DisplayPipelineStage')
doc_printer.save_documentation('DisplayPipelineStage', curr_active_pipeline.stage, non_expanded_item_keys=['_reverse_cellID_index_map', 'pf_listed_colormap', 'computation_results', 'active_configs', 'logger']) # 'Logger'
```


```python jupyter={"outputs_hidden": false} tags=[]
stage# doc_printer.reveal_output_files_in_system_file_manager()
```


```python jupyter={"outputs_hidden": false} tags=[]
filtered_context = curr_active_pipeline.filtered_contexts['maze1']
filtered_context.adding_context(collision_prefix='computation_params', comp_params_name=a_computation_config_name)
```

