# %% [markdown]
# # Purpose
# This notebook serves to contain the final, mostly user-level documentation for Spike3D

# %% [markdown]
# # Visualizations

# %% [markdown]
# #### All plots with `batch_perform_all_plots(curr_active_pipeline)`

# %%
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots

_out = batch_perform_all_plots(curr_active_pipeline=curr_active_pipeline, enable_neptune=False)

# %% [markdown]
# # 🖼️ `SpikeRaster2D`, `SpikeRaster3D`, and `Spike3DRasterWindowWidget`

# %% [markdown]
# ## Basics

# %%
# Create a new `SpikeRaster2D` instance using `_display_spike_raster_pyqtplot_2D` and capture its outputs:
active_2d_plot, active_3d_plot, spike_raster_window = curr_active_pipeline.plot._display_spike_rasters_pyqtplot_2D.values()

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### Getting the existing `Spike3DRasterWindowWidget`

# %%
from pyphoplacecellanalysis.GUI.Qt.SpikeRasterWindows.Spike3DRasterWindowWidget import Spike3DRasterWindowWidget

# Gets the existing SpikeRasterWindow or creates a new one if one doesn't already exist:
spike_raster_window, (active_2d_plot, active_3d_plot, main_graphics_layout_widget, main_plot_widget, background_static_scroll_plot_widget) = Spike3DRasterWindowWidget.find_or_create_if_needed(curr_active_pipeline)
spike_raster_window

# # Extras:
# active_2d_plot = spike_raster_window.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
# active_3d_plot = spike_raster_window.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
# main_graphics_layout_widget = active_2d_plot.ui.main_graphics_layout_widget # GraphicsLayoutWidget
# main_plot_widget = active_2d_plot.plots.main_plot_widget # PlotItem
# background_static_scroll_plot_widget = active_2d_plot.plots.background_static_scroll_window_plot # PlotItem

# %% [markdown]
# ### Changing display window
# `pyphoplacecellanalysis.General.Model.TimeWindow.TimeWindow`
# `SpikesDataframeWindow(LiveWindowedData)`

# %%
spikes_window = spike_raster_window.spikes_window # SpikesDataframeWindow; pyphoplacecellanalysis.General.Model.TimeWindow.TimeWindow
spikes_window.update_window_start_end(451.8908457518555, 451.9895490613999) ## Works but does not trigger refresh/update of the window. The changes are reflected as soon as you try to scroll at all though.


# %%
spikes_window.window_duration # Prints the current window's duration. The win. dur. label control in the left bar is not updated.

desired_window_fraction: float = 0.1 # 10% of the window is the default jump size
relevant_jump_duration: float = spikes_window.window_duration * desired_window_fraction
relevant_jump_duration


# %%
## Getting the current Display Window

# %%
spike_raster_window.total_data_start_time
spike_raster_window.total_data_end_time
spikes_window = spike_raster_window.spikes_window # SpikesDataframeWindow; pyphoplacecellanalysis.General.Model.TimeWindow.TimeWindow


# %%
active_2d_plot.total_data_start_time

# %% [markdown]
# ### Updating the display window start time programmatically

# %%
start_t = spike_raster_window.total_data_start_time
end_t = spike_raster_window.total_data_end_time
start_t, end_t
active_2d_plot.Render2DScrollWindowPlot_on_window_update(start_t, end_t)

# %%
total_fractional_seconds: float = 1244.907
spike_raster_window.update_animation(total_fractional_seconds)


# %% [markdown]
# ### Changing SpikeRaster2D/SpikeRaster3D neuron sort order:
# Enables sorting by pf1D pf peaks on the long or the short track:
# 2023-10-19

# %%
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.MultiContextComparingDisplayFunctions.LongShortTrackComparingDisplayFunctions import determine_long_short_pf1D_indicies_sort_by_peak

## Get 2D or 3D Raster from spike_raster_window
active_raster_plot = spike_raster_window.spike_raster_plt_2d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
if active_raster_plot is None:
	active_raster_plot = spike_raster_window.spike_raster_plt_3d # <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster.Spike2DRaster at 0x196c7244280>
	assert active_raster_plot is not None

# Sort the neurons by their peak on the long track AND on the short track:
included_unit_neuron_IDs = active_raster_plot.neuron_ids
new_active_2d_plotter_aclus_LONG_PEAK_sort_indicies = determine_long_short_pf1D_indicies_sort_by_peak(curr_active_pipeline=curr_active_pipeline, curr_any_context_neurons=included_unit_neuron_IDs, sortby=["long_pf_peak_x", "short_pf_peak_x", 'neuron_IDX']) # get the neuron_ids to be sorted from the raster plot
new_active_2d_plotter_aclus_SHORT_PEAK_sort_indicies = determine_long_short_pf1D_indicies_sort_by_peak(curr_active_pipeline=curr_active_pipeline, curr_any_context_neurons=included_unit_neuron_IDs, sortby=["short_pf_peak_x", "long_pf_peak_x", 'neuron_IDX']) # get the neuron_ids to be sorted from the raster plot

display(new_active_2d_plotter_aclus_LONG_PEAK_sort_indicies)
display(new_active_2d_plotter_aclus_SHORT_PEAK_sort_indicies)
# new_active_2d_plotter_aclus_sort_indicies # array([14,  3,  1,  2,  5,  9,  0, 20, 16, 24,  7, 19, 17, 21, 11, 10, 13, 12,  4, 18, 25,  6, 15, 23, 22,  8])


# %%
# Update the sort order on the Spike2DPlotter to align with the LONG TRACK pf1D field peaks:
active_raster_plot.unit_sort_order = new_active_2d_plotter_aclus_LONG_PEAK_sort_indicies

# Update the sort order on the Spike2DPlotter to align with the SHORT TRACK pf1D field peaks:
active_raster_plot.unit_sort_order = new_active_2d_plotter_aclus_SHORT_PEAK_sort_indicies

# Restore the original sort order of Spike2DPlotter:
original_neuron_plotter_aclus_sort_index = np.arange(len(new_active_2d_plotter_aclus_LONG_PEAK_sort_indicies))
active_raster_plot.unit_sort_order = original_neuron_plotter_aclus_sort_index


# %%
active_2d_plot.unit_sort_order = new_active_2d_plotter_aclus_sort_indicies

# %% [markdown]
# ### Working pretty cool lap plotter

# %%
from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import plot_lap_trajectories_2d
# Complete Version:
fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=len(curr_active_pipeline.sess.laps.lap_id), active_page_index=0)

# %%
# Paginated Version:
fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=22, active_page_index=0)
fig, axs, laps_pages = plot_lap_trajectories_2d(curr_active_pipeline.sess, curr_num_subplots=22, active_page_index=1)

# %% [markdown]
# ![image.png](attachment:image.png) ![image-2.png](attachment:image-2.png)

# %% [markdown]
# ## 📈 Rendered Time Curves Documentation Guide
# 
# #### `PyQtGraphSpecificTimeCurvesMixin(TimeCurvesViewMixin)`: mostly overriden for Spike2DRaster, but defines main plotting functions for Spike3DRaster
# pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.RenderTimeCurvesMixin.PyQtGraphSpecificTimeCurvesMixin
# 

# %%
add_3D_time_curves

# %%
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves import GeneralRenderTimeCurves
from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable import LocalMenus_AddRenderable

# %% [markdown]
# ### For `Spike2DRaster`

# %% [markdown]
# ##### Requires
# 
# `self.params.time_curves_no_update`
# 
# ### Single Datasource for time-curves:
# `self.params.time_curves_datasource`
# 
# 

# %% [markdown]
# #### Provides
# `self.ui.main_time_curves_view_widget`  
# `self.ui.main_time_curves_view_legend`  

# %% [markdown]
# #### Functions
# `clear_all_3D_time_curves(self)`  
# 
# `update_3D_time_curves(self)`  
# 
# `_build_or_update_time_curves_legend(self)`  
# 
# ---
# ##### `_build_or_update_time_curves_plot`: uses or builds a new `self.ui.main_time_curves_view_widget`, which the item is added to
# `_build_or_update_time_curves_plot(self, plot_name, points, **kwargs)`  
# 
# ---
# update_3D_time_curves_baseline_grid_mesh
# remove_3D_time_curves_baseline_grid_mesh
# 

# %% [markdown]
# #### TimeCurvesViewMixin/PyQtGraphSpecificTimeCurvesMixin specific overrides for 2D:
# """ 
# As soon as the first 2D Time Curve plot is needed, it creates:  
# 
#     `self.ui.main_time_curves_view_widget - PlotItem by calling add_separate_render_time_curves_plot_item(...)`  
# 
# 
# `main_time_curves_view_widget` creates new `PlotDataItems` by calling `self.ui.main_time_curves_view_widget.plot(...)`  
#     This `.plot(...)` command can take either:   
#         `.plot(x=x, y=y)`  
#         `.plot(ndarray(N,2)): single numpy array with shape (N, 2), where x=data[:,0] and y=data[:,1]`  
# 
# """

# %%


# %% [markdown]
# ### Procedure: Adding new Curves:
# 1. Copy `pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.SpecificTimeCurves.PositionRenderTimeCurves` into a new structure, changing as needed to display your desired variables
# 2. Add your new curve class to the import list at the top of `pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable` 
# 3. Use QtDesigner to add your menu in `GUI/Qt/Menus/LocalMenus_AddRenderable/LocalMenus_AddRenderable.ui` with an appropriate name.
# 	1. The objectName must follow the convention: `actionAddTimeCurves_Position`  -> e.g. `actionAddTimeCurves_Velocity`
# 4. Save and compile the .ui file (In VSCode: Right click > Compile .ui file)
# 5. Inside `LocalMenus_AddRenderable.build_renderable_menu(...)` add the appropriate entry to the `submenu_addTimeCurves` and `submenu_addTimeCurvesCallbacks` arrays.
# 	1. `lambda evt=None: VelocityRenderTimeCurves.add_render_time_curves(curr_sess=sess, destination_plot=destination_plot),`

# %% [markdown]
# ##### TODO: Time-curve adding improvements
# Enable users to 'register' new curves which are then added to the menu and the plot

# %% [markdown]
# ## Screenshots

# %% [markdown]
# ![[WithPBE_Epochs.png|500]]
# 
# ![image.png](attachment:image.png)

# %% [markdown]
# ## Dock Widgets and Tracks
# Docked widgets can be added in several places to SpikeRaster2D

# %% [markdown]
# ### Variables
# ```python
# self.ui.matplotlib_view_widgets
# ```

# %%
## Variables

# %%
def get_flat_dock_identifiers_list(self, debug_print=False) -> List[str]:

def get_flat_dockitems_list(self, debug_print=False) -> List[Dock]:
    
def get_flat_widgets_list(self, debug_print=False) -> List["QtWidgets.QWidget"]:
    

# %% [markdown]
# ### DynamicDockDisplayAreaContentMixin
# ```python
# def add_display_dock(self, identifier=None, widget=None, dockSize=(300,200), dockAddLocationOpts=['bottom'], display_config:CustomDockDisplayConfig=None, **kwargs) -> Tuple["QtWidgets.QWidget", Dock]:
# 
# def find_display_dock(self, identifier) -> Optional[Dock]:
# 
# def rename_display_dock(self, original_identifier, new_identifier):
# 
# def remove_display_dock(self, identifier):
# 
# 
# ```

# %% [markdown]
# # SpikeRaster2D Docked Widgets and "Tracks"

# %% [markdown]
# ```python
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import SynchronizedPlotMode
# 
# track_shapes_dock_identifier: str = 'TrackFrames2D'
# track_shapes_dock_items = spike_raster_plt_2d.add_new_matplotlib_render_plot_widget(name=track_shapes_dock_identifier, sync_mode=SynchronizedPlotMode.TO_WINDOW)
# track_shapes_dock_ts_widget, track_shapes_dock_fig, track_shapes_dock_ax_list, track_shapes_dock_item = track_shapes_dock_items
# track_shapes_dock_track_ax = track_shapes_dock_ax_list[0]
# 
# ## sync up the widgets
# # spike_raster_plt_2d.sync_matplotlib_render_plot_widget(track_shapes_dock_identifier, sync_mode=SynchronizedPlotMode.TO_WINDOW)
# ```

# %%
active_2d_plot.add_docked_marginal_track(...)



# %% [markdown]
# time_bin_size = epochs_decoding_time_bin_size
# info_string: str = f" - t_bin_size: {time_bin_size}"
# identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_marginal_track(name='non-PBE_marginal_over_track_ID',
#                                                                                         time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID, extended_dock_title_info=info_string)
# 

# %%
@function_attributes(short_name=None, tags=['IMPORTANT', 'FINAL', 'track', 'posterior', '1D'], input_requires=[], output_provides=[], uses=[], used_by=['add_docked_decoded_posterior_track_from_result'], creation_date='2025-03-21 08:32', related_items=[])
def add_docked_decoded_posterior_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None, should_defer_render:bool=False, **kwargs):

def add_docked_decoded_posterior_slices_track(self, name: str, slices_time_window_centers: List[NDArray], slices_posteriors: List[NDArray], xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None, should_defer_render:bool=False, **kwargs):
    
@function_attributes(short_name=None, tags=['IMPORTANT', 'FINAL', 'track', 'posterior'], input_requires=[], output_provides=[], uses=['add_docked_decoded_posterior_track'], used_by=[], creation_date='2025-03-21 08:10', related_items=[])
def add_docked_decoded_posterior_track_from_result(self, name: str, a_1D_decoded_result: Union[SingleEpochDecodedResult, DecodedFilterEpochsResult], xbin: Optional[NDArray]=None, measured_position_df: Optional[pd.DataFrame]=None, **kwargs):
    """ adds a decoded 1D posterior from a decoded result """
        
    
@function_attributes(short_name=None, tags=['IMPORTANT', 'track', 'posterior', 'marginal'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-03-21 08:10', related_items=[])
def add_docked_marginal_track(self, name: str, time_window_centers: NDArray, a_1D_posterior: NDArray, xbin: Optional[NDArray]=None, a_variable_name: Optional[str]=None, a_dock_config: Optional[CustomDockDisplayConfig]=None, extended_dock_title_info: Optional[str]=None):
    

# %%
time_bin_size = epochs_decoding_time_bin_size
info_string: str = f" - t_bin_size: {time_bin_size}"
identifier_name, widget, matplotlib_fig, matplotlib_fig_axes, dock_item = active_2d_plot.add_docked_marginal_track(name='non-PBE_marginal_over_track_ID',
                                                                                        time_window_centers=time_window_centers, a_1D_posterior=non_PBE_marginal_over_track_ID, extended_dock_title_info=info_string)


# %% [markdown]
# ## Removing Docks:
# ```python
# spike_raster_plt_2d.remove_display_dock(identifier='Frames2D')
# ```
# 
# 

# %% [markdown]
# ### `DynamicDockDisplayAreaOwningMixin`    
# ```python
# @property 
# def dock_manager_widget(self) -> DynamicDockDisplayAreaContentMixin
#     
# def find_display_dock(self, identifier) -> Optional[Dock]
#     
# def add_display_dock(self, identifier=None, widget=None, dockSize=(300,200), dockAddLocationOpts=['bottom'], **kwargs)
#     
# def remove_display_dock(self, identifier)
#     
# def rename_display_dock(self, original_identifier, new_identifier)
#     
# def clear_all_display_docks(self)
# 
# def get_flat_dock_identifiers_list(self, debug_print=False) -> List[str]
# 
# def get_flat_dockitems_list(self, debug_print=False) -> List[Dock]
# 
# def get_flat_widgets_list(self, debug_print=False) -> List["QtWidgets.QWidget"]
# 
# def get_flat_dock_item_tuple_dict(self, debug_print=False) -> Dict[str, Tuple[Dock, Optional["QtWidgets.QWidget"]]]
# 
# def get_dockGroup_dock_dict(self, debug_print=False) -> Dict[str, List[Dock]]

# %% [markdown]
# ## `matplotlib_view_widget`

# %% [markdown]
# ### Dynamic Matplotlib Plots in Spike2DRaster

# %% [markdown]
# `self.ui.matplotlib_view_widget`
# 
# In `Spike2DRaster`
# ```python
# self.ui.dynamic_docked_widget_container = NestedDockAreaWidget()
# ```
# Helper Functions:
# ```python
# # matplotlib render subplot __________________________________________________________________________________________ #
#     def add_new_matplotlib_render_plot_widget(self, row=1, col=0, name='matplotlib_view_widget'):
#         """ creates a new MatplotlibTimeSynchronizedWidget, a container widget that holds a matplotlib figure, and adds it as a row to the main layout """
# 
#     def remove_matplotlib_render_plot_widget(self):
#         """ removes the subplot - does not work yet """
# 
#     def sync_matplotlib_render_plot_widget(self):
#         """ Perform Initial (one-time) update from source -> controlled: """
# 
#     def clear_all_matplotlib_plots(self):
#         """ required by the menu function """
# 
#     curr_widget, curr_fig, curr_ax = active_2d_plot.find_matplotlib_render_plot_widget('Custom Decoder')
#         
# ```

# %% [markdown]
# ### Adding PyQtGraph-based independent raster plot as track
# `_raster_tracks_out_dict = active_2d_plot.prepare_pyqtgraph_rasterPlot_track(name_modifier_suffix='test')`

# %% [markdown]
# # ◽📣🟧 Rectangle Epoch Documentation Guide

# %% [markdown]
# 
# <!-- C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Model\Datasources\IntervalDatasource.py -->
# <!-- [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/Datasources/IntervalDatasource.py:20](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/Datasources/IntervalDatasource.py:20) -->
# [IntervalsDatasource](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Model/Datasources/IntervalDatasource.py:20)
# 
# <!-- C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\EpochRenderingMixin.py -->
# [EpochRenderingMixin](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/EpochRenderingMixin.py:42)
# 
# <!-- C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\PyQtPlot\Widgets\Mixins\RenderTimeEpochs\RenderTimeEpoch3DMeshesMixin.py -->
# [RenderTimeEpoch3DMeshesMixin](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/RenderTimeEpoch3DMeshesMixin.py:20)
# 
# #### `Render2DEventRectanglesHelper`:
# 
# [GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/Render2DEventRectanglesHelper.py](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/Render2DEventRectanglesHelper.py:30)
# 
# #### `Specific2DRenderTimeEpochs`:
# 
# [GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/Specific2DRenderTimeEpochs.py](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/Specific2DRenderTimeEpochs.py:24)
# 
# 
# #### `Spike2DRaster`
# _perform_add_render_item
# _perform_remove_render_item
# add_laps_intervals/remove_laps_intervals
# add_PBEs_intervals/remove_PBEs_intervals
# 
# 

# %% [markdown]
# ## Screenshots

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# ### 3D Interval Rects
# #rectangles #IntervalRectsItem #interval #PBEs #3d #spike3d 
# 
# 
# Here you can see many short intervals rendered as cyan rectangles on the floor of the 3D Raster

# %% [markdown]
# `active_3d_plot.add_rendered_intervals(new_ripples_intervals_datasource, name='new_ripples')`

# %% [markdown]
# ![python_JwdIMVHpEQ.png](attachment:52498aab-31a8-4a0b-8add-0728809de9ab.png)
# ![image.png](attachment:dabc70cf-76b1-45b6-b7a0-a3bf785e5391.png)

# %% [markdown]
# ## ◽📣 ✅ Testing 2D Rectangle Epochs on Raster Plot

# %%
laps_interval_datasource = Specific2DRenderTimeEpochsHelper.build_Laps_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=max_series_top, series_height=1.0) # series_vertical_offset=42.0
new_PBEs_interval_datasource = Specific2DRenderTimeEpochsHelper.build_PBEs_render_time_epochs_datasource(curr_sess=sess, series_vertical_offset=(max_series_top+1.0), series_height=3.0) # new_PBEs_interval_datasource

## General Adding:
active_2d_plot.add_rendered_intervals(new_PBEs_interval_datasource, name='PBEs', child_plots=[background_static_scroll_plot_widget, main_plot_widget], debug_print=False)
active_2d_plot.add_rendered_intervals(laps_interval_datasource, name='Laps', child_plots=[background_static_scroll_plot_widget, main_plot_widget], debug_print=False)

# %%
active_2d_plot.add_laps_intervals(sess)

# %%
active_2d_plot.remove_laps_intervals()

# %%
# active_2d_plot.add_PBEs_intervals(sess)

# %%
active_2d_plot.interval_rendering_plots

# %%
active_2d_plot.clear_all_rendered_intervals()

# %%
interval_info = active_2d_plot.list_all_rendered_intervals()
interval_info

# %% [markdown]
# ## 📣 Programmatically adding several epoch rectangles by calling the addRenderable context menu functions all at once for SpikeRaster2D

# %%
add_renderables_menu = active_2d_plot.ui.menus.custom_context_menus.add_renderables[0].programmatic_actions_dict
menu_commands = ['AddTimeIntervals.PBEs', 'AddTimeIntervals.Ripples', 'AddTimeIntervals.Replays', 'AddTimeIntervals.Laps', 'AddTimeIntervals.SessionEpochs']
for a_command in menu_commands:
    add_renderables_menu[a_command].trigger()

# %% [markdown]
# ## ◽📣 Updating Epochs visual appearance

# %%
interval_info = active_2d_plot.list_all_rendered_intervals()
interval_info

# %%
active_2d_plot.clear_all_rendered_intervals()

# %%
active_2d_plot.interval_rendering_plots

# %%
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

# %% [markdown]
# ### Concise Update:

# %%
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


# %% [markdown]
# ### Build Stacked Layout:

# %%
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin, RenderedEpochsItemsContainer

rendered_interval_keys = list(interval_info.keys())
desired_interval_height_ratios = [2.0, 2.0, 1.0, 0.1, 1.0, 1.0, 1.0] # ratio of heights to each interval
required_vertical_offsets, required_interval_heights = EpochRenderingMixin.build_stacked_epoch_layout(desired_interval_height_ratios, epoch_render_stack_height=20.0, interval_stack_location='below')
stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(rendered_interval_keys, required_vertical_offsets, required_interval_heights)} # Build a stacked_epoch_layout_dict to update the display
active_2d_plot.update_rendered_intervals_visualization_properties(stacked_epoch_layout_dict)

# %% [markdown]
# ## ◽📣 Building Epochs to render from scratch from a list of pd.DataFrames and custom-defined properties describing their visual appearance

# %%
from pyphocorehelpers.Filesystem.HDF5.hdf5_file_helpers import hdf5_to_pandas_df_dict

debug_output_hdf5_file_path = Path('output', 'laps_train_test_split.h5').resolve()
assert debug_output_hdf5_file_path.exists()
loaded_hdf5_laps_data_dict, failed_keys = hdf5_to_pandas_df_dict(hdf5_path=debug_output_hdf5_file_path)

## Extract the specific results:
decoder_name: str = 'long_LR'
laps_df = loaded_hdf5_laps_data_dict[f'/provided/{decoder_name}/laps_df']
train_df = loaded_hdf5_laps_data_dict[f'/valid/{decoder_name}/train_df']
test_df = loaded_hdf5_laps_data_dict[f'/valid/{decoder_name}/test_df']


from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, inline_mkColor
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.EpochRenderingMixin import EpochRenderingMixin, RenderedEpochsItemsContainer
from pyphoplacecellanalysis.General.Model.Datasources.IntervalDatasource import IntervalsDatasource
from neuropy.utils.mixins.time_slicing import TimeColumnAliasesProtocol


## Use the three dataframes as separate Epoch series:
train_test_split_laps_dfs_dict = {
    'LapsAll': laps_df,
    'LapsTrain': train_df,
    'LapsTest': test_df,
}

train_test_split_laps_epochs_formatting_dict = {
    'LapsAll':dict(y_location=-10.0, height=7.5, pen_color=inline_mkColor('white', 0.8), brush_color=inline_mkColor('white', 0.5)),
    'LapsTrain':dict(y_location=-2.0, height=1.5, pen_color=inline_mkColor('purple', 0.8), brush_color=inline_mkColor('purple', 0.5)),
    'LapsTest':dict(y_location=-12.0, height=1.5, pen_color=inline_mkColor('green', 0.8), brush_color=inline_mkColor('green', 0.5)),
}

required_vertical_offsets, required_interval_heights = EpochRenderingMixin.build_stacked_epoch_layout([0.2, 1.0, 1.0], epoch_render_stack_height=40.0, interval_stack_location='below') # ratio of heights to each interval
stacked_epoch_layout_dict = {interval_key:dict(y_location=y_location, height=height) for interval_key, y_location, height in zip(list(train_test_split_laps_epochs_formatting_dict.keys()), required_vertical_offsets, required_interval_heights)} # Build a stacked_epoch_layout_dict to update the display
# stacked_epoch_layout_dict # {'LapsAll': {'y_location': -3.6363636363636367, 'height': 3.6363636363636367}, 'LapsTrain': {'y_location': -21.818181818181817, 'height': 18.18181818181818}, 'LapsTest': {'y_location': -40.0, 'height': 18.18181818181818}}

# replaces 'y_location', 'position' for each dict:
train_test_split_laps_epochs_formatting_dict = {k:(v|stacked_epoch_layout_dict[k]) for k, v in train_test_split_laps_epochs_formatting_dict.items()}
train_test_split_laps_epochs_formatting_dict

# OUTPUTS: train_test_split_laps_dfs_dict, train_test_split_laps_epochs_formatting_dict

# %%
## INPUTS: train_test_split_laps_dfs_dict
train_test_split_laps_dfs_dict = {k:TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df=v, required_columns_synonym_dict=IntervalsDatasource._time_column_name_synonyms) for k, v in train_test_split_laps_dfs_dict.items()}

## Build interval datasources for them:
train_test_split_laps_dfs_datasources_dict = {k:General2DRenderTimeEpochs.build_render_time_epochs_datasource(v) for k, v in train_test_split_laps_dfs_dict.items()}
## INPUTS: active_2d_plot, train_test_split_laps_epochs_formatting_dict, train_test_split_laps_dfs_datasources_dict
assert len(train_test_split_laps_epochs_formatting_dict) == len(train_test_split_laps_dfs_datasources_dict)
for k, an_interval_ds in train_test_split_laps_dfs_datasources_dict.items():
    an_interval_ds.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, **(train_test_split_laps_epochs_formatting_dict[k] | kwargs)))

## Full output: train_test_split_laps_dfs_datasources_dict

# %%
# actually add the epochs:
for k, an_interval_ds in train_test_split_laps_dfs_datasources_dict.items():
    active_2d_plot.add_rendered_intervals(an_interval_ds, name=f'{k}', debug_print=False) # adds the interval



# %%
## They can later be updated via:
active_2d_plot.update_rendered_intervals_visualization_properties(train_test_split_laps_epochs_formatting_dict)


# %%
## INPUTS: train_test_split_laps_dfs_dict
train_test_split_laps_dfs_dict = {k:TimeColumnAliasesProtocol.renaming_synonym_columns_if_needed(df=v, required_columns_synonym_dict=IntervalsDatasource._time_column_name_synonyms) for k, v in train_test_split_laps_dfs_dict.items()}

## Build interval datasources for them:
train_test_split_laps_dfs_datasources_dict = {k:General2DRenderTimeEpochs.build_render_time_epochs_datasource(v) for k, v in train_test_split_laps_dfs_dict.items()}
## INPUTS: active_2d_plot, train_test_split_laps_epochs_formatting_dict, train_test_split_laps_dfs_datasources_dict
assert len(train_test_split_laps_epochs_formatting_dict) == len(train_test_split_laps_dfs_datasources_dict)
for k, an_interval_ds in train_test_split_laps_dfs_datasources_dict.items():
    an_interval_ds.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, **(train_test_split_laps_epochs_formatting_dict[k] | kwargs)))

## Full output: train_test_split_laps_dfs_datasources_dict

# %%
# actually add the epochs:
for k, an_interval_ds in train_test_split_laps_dfs_datasources_dict.items():
    active_2d_plot.add_rendered_intervals(an_interval_ds, name=f'{k}', debug_print=False) # adds the interval



# %%
## They can later be updated via:
active_2d_plot.update_rendered_intervals_visualization_properties(train_test_split_laps_epochs_formatting_dict)


# %% [markdown]
# ## ◽📣 Get list of existing interval rect datasources:
# blah blah

# %%
# Plot items:
active_2d_plot.interval_rendering_plots

# %%
# active_2d_plot.interval_datasources.new_ripples
interval_info = active_2d_plot.list_all_rendered_intervals()
interval_info

# %%
active_2d_plot.interval_datasources # RenderPlotsData
# datasource_to_update

# %%
active_2d_plot.interval_datasources.PBEs # IntervalsDatasource

# %% [markdown]
# ## ◽📣 Update existing interval rects:
# Write a function that takes your existing datasource dataframe and updates its columns.

# %% [markdown]
# ### Before Update:
# ![python_YwFQ3gs3K2.png](attachment:d13ff2db-bf10-457f-8184-9cf4f822eb38.png)

# %%
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

# %%
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.RenderTimeEpochs.Specific2DRenderTimeEpochs import General2DRenderTimeEpochs, inline_mkColor
## Inline Concise: Position Replays, PBEs, and Ripples all below the scatter:
active_2d_plot.interval_datasources.Replays.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-10.0, height=7.5, pen_color=inline_mkColor('orange', 0.8), brush_color=inline_mkColor('orange', 0.5), **kwargs)) ## Fully inline
active_2d_plot.interval_datasources.PBEs.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-2.0, height=1.5, pen_color=inline_mkColor('pink', 0.8), brush_color=inline_mkColor('pink', 0.5), **kwargs)) ## Fully inline
active_2d_plot.interval_datasources.Ripples.update_visualization_properties(lambda active_df, **kwargs: General2DRenderTimeEpochs._update_df_visualization_columns(active_df, y_location=-12.0, height=1.5, pen_color=inline_mkColor('cyan', 0.8), brush_color=inline_mkColor('cyan', 0.5), **kwargs)) ## Fully inline

# %% [markdown]
# ### Post Update:
# ![python_LKGNtQCkQH.png](attachment:99906b90-2fdd-42ec-8536-ec0e52b73c68.png)

# %%
datasource_to_update.custom_datasource_name

# %%
datasource_to_update.df

# %%
spike_raster_window.spike_raster_plt_2d.add_rendered_intervals(datasource_to_update, name='CustomRipples', debug_print=True) 

# %%
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

# %%
plots.fig.show()

# %%
# laps_position_times_list = [np.squeeze(lap_pos_df[['t']].to_numpy()) for lap_pos_df in lap_specific_position_dfs]
# laps_position_traces_list = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]

# epochs = sess.laps.to_dataframe()
# epoch_slices = epochs[['start', 'stop']].to_numpy()
# epoch_description_list = [f'lap {epoch_tuple.lap_id} (maze: {epoch_tuple.maze_id}, direction: {epoch_tuple.lap_dir})' for epoch_tuple in epochs[['lap_id','maze_id','lap_dir']].itertuples()]
# print(f'epoch_description_list: {epoch_description_list}')


from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import stacked_epoch_slices_view

stacked_epoch_slices_view_laps_containers = stacked_epoch_slices_view(epoch_slices, laps_position_times_list, laps_position_traces_list, name=f'stacked_epoch_slices_view_new_ripples: shank {shank_id}')
params, plots_data, plots, ui = stacked_epoch_slices_view_laps_containers

# %%
doubleSpinBox_ActiveWindowStartTime
doubleSpinBox_ActiveWindowEndTime

# %%
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\old_global_computation_results.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\loadedSessPickle.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\global_computation_results_2023-10-06.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\global_computation_results_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\global_computation_results.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\backup-20231113092010-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\backup-20231110234635-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\20231113095949-loadedSessPickle.pkl


# %% [markdown]
# ## ◽📣 Removing/Clearing existing interval rects:

# %% [markdown]
# ### Selectively Removing:

# %%
active_2d_plot.remove_rendered_intervals(name='PBEs', child_plots_removal_list=[main_plot_widget]) # Tests removing a single series from a single plot (main_plot_widget)
active_2d_plot.remove_rendered_intervals(name='PBEs') # Tests removing a single series ('PBEs') from all plots it's on

# %% [markdown]
# # 3D (PyVista/Vedo/etc)-based plots:

# %%
curr_active_pipeline.display('_display_3d_interactive_spike_and_behavior_browser', active_config_name) # this works now!

# %%
display_dict = curr_active_pipeline.display('_display_3d_interactive_custom_data_explorer', active_config_name) # does not work, missing color info?
iplapsDataExplorer = display_dict['iplapsDataExplorer']
# plotter is available at
p = display_dict['plotter']
iplapsDataExplorer

# %%
# curr_kdiba_pipeline.display(DefaultDisplayFunctions._display_3d_interactive_custom_data_explorer, 'maze1') # works!
curr_active_pipeline.display('_display_3d_interactive_tuning_curves_plotter', 'maze1_PYR') # works!

# %%


# %% [markdown]
# ### Adjusting Spike Emphasis:
# #### Usage Examples:
# ```python
# from pyphoplacecellanalysis.General.Mixins.SpikesRenderingBaseMixin import SpikeEmphasisState
# 
# ## Example 1: De-emphasize spikes excluded from the placefield calculations:
# is_spike_included_in_pf = np.isin(spike_raster_window.spike_raster_plt_2d.spikes_df.index, active_pf_2D.filtered_spikes_df.index)
# spike_raster_window.spike_raster_plt_2d.update_spike_emphasis(np.logical_not(is_spike_included_in_pf), SpikeEmphasisState.Deemphasized)
# 
# ## Example 2: De-emphasize spikes that don't have their 'aclu' from a given set of indicies:
# is_spike_included = spike_raster_window.spike_raster_plt_2d.spikes_df.aclu.to_numpy() == 2
# spike_raster_window.spike_raster_plt_2d.update_spike_emphasis(np.logical_not(is_spike_included), SpikeEmphasisState.Deemphasized)
# 
# ## Example 3: De-emphasize all spikes 
# active_2d_plot.update_spike_emphasis(new_emphasis_state=SpikeEmphasisState.Deemphasized)
# 
# ## Example 4: Hide all spikes entirely
# active_2d_plot.update_spike_emphasis(new_emphasis_state=SpikeEmphasisState.Hidden)
# ```
# 
# #### Notes
# Looks like there is very advanced emphasis functionality that I haven't explored. See Code example below:
# ```python
# 
# # SpikeEmphasisState
# state_alpha = {SpikeEmphasisState.Hidden: 0.01,
# 			   SpikeEmphasisState.Deemphasized: 0.1,
# 			   SpikeEmphasisState.Default: 0.5,
# 			   SpikeEmphasisState.Emphasized: 1.0,
# }
# 
# # state_color_adjust_fcns: functions that take the base color and call build_adjusted_color to get the adjusted color for each state
# state_color_adjust_fcns = {SpikeEmphasisState.Hidden: lambda x: build_adjusted_color(x),
# 			   SpikeEmphasisState.Deemphasized: lambda x: build_adjusted_color(x, saturation_scale=0.35, value_scale=0.8),
# 			   SpikeEmphasisState.Default: lambda x: build_adjusted_color(x),
# 			   SpikeEmphasisState.Emphasized: lambda x: build_adjusted_color(x, value_scale=1.25),
# }
# 
# ```

# %% [markdown]
# ### Assigning Cell Colors

# %% [markdown]
# 2023-10-18 - Use `build_cell_colors` to build configs
# 

# %%

@function_attributes(short_name=None, tags=['colors', 'neuron_identity'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-18 11:33', related_items=[])
def build_cell_colors(n_neurons:int, colormap_name='hsv', colormap_source='matplotlib'):
	"""Cell Colors from just n_neurons using pyqtgraph colormaps.
	
	"""
	cm = pg.colormap.get(colormap_name, source=colormap_source) # prepare a linear color map

	# unit_colors_list = None # default rainbow of colors for the raster plots
	neuron_qcolors_list = cm.mapToQColor(np.arange(n_neurons)/float(n_neurons-1)) # returns a list of QColors
	neuron_colors_ndarray = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=True)
	# neuron_colors_ndarray = DataSeriesColorHelpers.qColorsList_to_NDarray(neuron_qcolors_list, is_255_array=False)
	return neuron_qcolors_list, neuron_colors_ndarray


included_unit_neuron_IDs = active_2d_plot.neuron_ids
# n_neurons = len(EITHER_subset.track_exclusive_aclus)
n_neurons = len(included_unit_neuron_IDs)
# neuron_qcolors_list, neuron_colors_ndarray = build_cell_colors(n_neurons)
# neuron_qcolors_list, neuron_colors_ndarray = build_cell_colors(n_neurons, colormap_name='gist_rainbow')
neuron_qcolors_list, neuron_colors_ndarray = build_cell_colors(n_neurons, colormap_name='PAL-relaxed_bright', colormap_source=None)


# %% [markdown]
# ## Adding new Custom Menu Commands:
# #menu #SpikeRaster2D

# %%
from pyphoplacecellanalysis.GUI.Qt.Menus.BaseMenuProviderMixin import BaseMenuCommand

@define(slots=False)
class AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand(BaseMenuCommand):
	""" 2024-01-17 
	Adds four rows to the SpikeRaster2D showing the continuously decoded posterior for each of the four 1D decoders

	Usage:
	from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand

	"""
	_spike_raster_window = field()
	_active_pipeline = field(alias='curr_active_pipeline')
	_active_config_name = field(default=None)
	_context = field(default=None, alias="active_context")
	_display_output = field(default=Factory(dict))

	@classmethod
	def _perform_add_new_decoded_row(cls, curr_active_pipeline, active_2d_plot, a_dock_config, a_decoder_name: str, a_decoder, a_decoded_result=None):
		""" adds a single decoded row to the matplotlib dynamic output
		
		# a_decoder_name: str = "long_LR"

		"""
		from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DecoderPredictionError import plot_1D_most_likely_position_comparsions
		
		## ✅ Add a new row for each of the four 1D directional decoders:
		identifier_name: str = f'{a_decoder_name}_ContinuousDecode'
		print(f'identifier_name: {identifier_name}')
		widget, matplotlib_fig, matplotlib_fig_axes = active_2d_plot.add_new_matplotlib_render_plot_widget(name=identifier_name, dockSize=(65, 200), display_config=a_dock_config)
		an_ax = matplotlib_fig_axes[0]

		# _active_config_name = None
		variable_name: str = a_decoder_name
		active_decoder = deepcopy(a_decoder)
		
		if a_decoded_result is not None:
			active_result = deepcopy(a_decoded_result) # already decoded
			assert (active_result.num_filter_epochs == 1), f"currently only supports decoded results (DecodedFilterEpochsResult) computed with a single epoch for all time bins, but active_result.num_filter_epochs: {active_result.num_filter_epochs}"
			active_marginals = active_result.marginal_x_list[0]
		else:
			# no previously decoded result, fallback to the decoder's internal properties        
			active_marginals = active_decoder.marginal.x
			

		active_bins = active_decoder.xbin

		# active_most_likely_positions = active_marginals.most_likely_positions_1D # Raw decoded positions
		active_most_likely_positions = None
		active_posterior = active_marginals.p_x_given_n

		# most_likely_positions_mode: 'standard'|'corrected'
		# fig, curr_ax = curr_active_pipeline.display('_display_plot_marginal_1D_most_likely_position_comparisons', _active_config_name, variable_name='x', most_likely_positions_mode='corrected', ax=an_ax) # ax=active_2d_plot.ui.matplotlib_view_widget.ax
		## Actual plotting portion:
		fig, curr_ax = plot_1D_most_likely_position_comparsions(None, time_window_centers=active_decoder.time_window_centers, xbin=active_bins,
																posterior=active_posterior,
																active_most_likely_positions_1D=active_most_likely_positions,
																ax=an_ax, variable_name=variable_name, debug_print=True, enable_flat_line_drawing=False)

		widget.draw() # alternative to accessing through full path?
		active_2d_plot.sync_matplotlib_render_plot_widget(identifier_name) # Sync it with the active window:
		return identifier_name, widget, matplotlib_fig, matplotlib_fig_axes

	@classmethod
	def add_directional_decoder_decoded_epochs(cls, curr_active_pipeline, active_2d_plot, debug_print=False):
		""" adds the decoded epochs for the long/short decoder from the global_computation_results as new matplotlib plot rows. """
		from pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig import DisplayColorsEnum
		from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.DynamicDockDisplayAreaContent import CustomDockDisplayConfig
		from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.RankOrderComputations import RankOrderAnalyses
		
		showCloseButton = True
		dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=showCloseButton), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=showCloseButton),
						CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=showCloseButton), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=showCloseButton))))


		## Uses the `global_computation_results.computed_data['DirectionalDecodersDecoded']`
		directional_decoders_decode_result: DirectionalDecodersDecodedResult = curr_active_pipeline.global_computation_results.computed_data['DirectionalDecodersDecoded']
		all_directional_pf1D_Decoder_dict: Dict[str, BasePositionDecoder] = directional_decoders_decode_result.pf1D_Decoder_dict
		# continuously_decoded_result_cache_dict = directional_decoders_decode_result.continuously_decoded_result_cache_dict
		time_bin_size: float = directional_decoders_decode_result.most_recent_decoding_time_bin_size
		print(f'time_bin_size: {time_bin_size}')
		continuously_decoded_dict: Dict[str, DecodedFilterEpochsResult] = directional_decoders_decode_result.most_recent_continuously_decoded_dict
		all_directional_continuously_decoded_dict = continuously_decoded_dict or {}

		# Need all_directional_pf1D_Decoder_dict
		output_dict = {}

		for a_decoder_name, a_decoder in all_directional_pf1D_Decoder_dict.items():
			a_dock_config = dock_configs[a_decoder_name]
			a_decoded_result = all_directional_continuously_decoded_dict.get(a_decoder_name, None) # already decoded
			_out_tuple = cls._perform_add_new_decoded_row(curr_active_pipeline=curr_active_pipeline, active_2d_plot=active_2d_plot, a_dock_config=a_dock_config, a_decoder_name=a_decoder_name, a_decoder=a_decoder, a_decoded_result=a_decoded_result)
			# identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = _out_tuple
			output_dict[a_decoder_name] = _out_tuple

		return output_dict


	def validate_can_display(self) -> bool:
		""" returns True if the item is enabled, otherwise returns false """
		try:
			curr_active_pipeline = self._active_pipeline
			# assert curr_active_pipeline is not None
			if curr_active_pipeline is None:
				raise ValueError("Current active pipeline is None!")
			active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
			# assert active_2d_plot is not None
			if active_2d_plot is None:
				raise ValueError("active_2d_plot is None!")

			return DirectionalDecodersDecodedResult.validate_has_directional_decoded_continuous_epochs(curr_active_pipeline=curr_active_pipeline)
			
		except Exception as e:
			print(f'Exception {e} occured in validate_can_display(), returning False')
			return False

	def execute(self, *args, **kwargs) -> None:
		## To begin, the destination plot must have a matplotlib widget plot to render to:
		# print(f'AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand.execute(...)')
		active_2d_plot = self._spike_raster_window.spike_raster_plt_2d
		# If no plot to render on, do this:
		output_dict = self.add_directional_decoder_decoded_epochs(self._active_pipeline, active_2d_plot) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
		# Update display output dict:
		for a_decoder_name, an_output_tuple in output_dict.items():
			identifier_name, widget, matplotlib_fig, matplotlib_fig_axes = an_output_tuple
			self._display_output[identifier_name] = an_output_tuple

		print(f'\t AddNewDirectionalDecodedEpochs_MatplotlibPlotCommand.execute() is done.')



# %% [markdown]
# ## 3D Tracks/Placefield/Spikes Visualizations

# %% [markdown]
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/placefield_plotting_mixins.py:97](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/placefield_plotting_mixins.py:97)
# ```python
# # From `PhoPositionalData.plotting.mixins.placefield_plotting_mixins.plot_placefields`
# self.params.should_override_disable_smooth_shading = True # if True, forces smooth_shading to be False regardless of other parameters    
# _temp_input_params = get_dict_subset(self.params, ['should_use_normalized_tuning_curves','should_pdf_normalize_manually','should_nan_non_visited_elements','should_force_placefield_custom_color','should_display_placefield_points', 'should_override_disable_smooth_shading', 'nan_opacity'])
# # print(f'_temp_input_params: {_temp_input_params}')
# 
# self.p, self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor'], temp_plots_data = plot_placefields2D(self.p, self.params.active_epoch_placefields, self.params.pf_colors, zScalingFactor=self.params.zScalingFactor, show_legend=self.params.show_legend, **_temp_input_params) # note that the get_dict_subset(...) thing is just a safe way to get only the relevant members.
# # Build the widget labels:
# self.params.unit_labels = temp_plots_data['unit_labels'] # fetch the unit labels from the extra data dict.
# self.params.pf_fragile_linear_neuron_IDXs = temp_plots_data['good_placefield_neuronIDs'] # fetch the unit labels from the extra data dict.
# ## TODO: For these, we actually want the placefield value as the Z-positions, will need to unwrap them or something (maybe .ravel(...)?)
# ## TODO: also need to add in the checkbox functionality to hide/show only the spikes for the highlighted units
# # .threshold().elevation()
# 
# ## Legend data:
# self.plots_data['tuningCurvePlotLegendData'] = temp_plots_data['legend_entries']
# ```

# %%
self.p, self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor'], temp_plots_data = plot_placefields2D(self.p, self.params.active_epoch_placefields, self.params.pf_colors, zScalingFactor=self.params.zScalingFactor, show_legend=self.params.show_legend, **_temp_input_params) # note that the get_dict_subset(...) thing is just a safe way to get only the relevant members.

# %%
update_plotColorsPlacefield2D(self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], neuron_id_color_update_dict=neuron_id_color_update_dict)


# %% [markdown]
# # 🏻‍💻 DEVELOPER SECTION

# %% [markdown]
# ## TODO/PENDING

# %% [markdown]
# #### Set the tooltip for each individual rect so that when you hover a rect it shows relevant information (that intervals series name, its (start, end) times, duration, index, etc
# search code in cell below to find where it's set generally. Try `CustomIntervalRectsItem`
# ![image.png](attachment:c06368e0-a801-4e50-8c41-dbe9eb6d50eb.png)

# %%
# Build the rendered interval item:
new_interval_rects_item = Render2DEventRectanglesHelper.build_IntervalRectsItem_from_interval_datasource(interval_datasource)
new_interval_rects_item.setToolTip(name) # The tooltip is set generically here to 'PBEs', 'Replays' or whatever the dataseries name is

# %% [markdown]
# ## 2023-04-13 - Desired Plotting Interface Idea:
# 

# %% [markdown]
# ```python
# # Rows of plots can be constructed trivially through lists:
# row_of_plots = [pg.plot(curr_cell_pf_curve, label='curr_cell_pf_curve'), pg.plot(curr_random_not_firing_cell_pf_curve, label='curr_random_not_firing_cell_pf_curve'), ...] 
# 	# I'd guess behind the scenes they would be converted into a helper.row([...]) object
# 
# # If you want a column instead, use helper.column
# column_of_plots = helper.column([pg.plot(curr_cell_pf_curve, label='curr_cell_pf_curve'), pg.plot(curr_random_not_firing_cell_pf_curve, label='curr_random_not_firing_cell_pf_curve'), ...])
# 
# # The returned objects are composable:
# row_of_layouts = [row_of_plots, column_of_plots] # stacks the layout objects just like they were plot objects
# 
# # Showing the result is easy, as is combining separate results in a new place:
# whole_figure_window = [row_of_layouts] 
# whole_figure_window.show()
# ```
# 
# """ 
# Relevant Functions:
# `perform_full_session_leave_one_out_decoding_analysis`:
# 	`perform_leave_one_aclu_out_decoding_analysis`:	from pyphoplacecellanalysis.Analysis.Decoder.decoder_result import perform_leave_one_aclu_out_decoding_analysis
# 	`_analyze_leave_one_out_decoding_results`: from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.DefaultComputationFunctions import _analyze_leave_one_out_decoding_results
# """
# 
# 

# %% [markdown]
# ## ⚓💯 Custom Container Classes for UI
# 

# %% [markdown]
# ### GenericMatplotlibContainer, GenericPyQtGraphContainer, PhoBaseContainerTool
# from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer, GenericPyQtGraphContainer, PhoBaseContainerTool

# %%
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool import GenericMatplotlibContainer, GenericPyQtGraphContainer, PhoBaseContainerTool

# %%


# %% [markdown]
# ### VisualizationParameters, RenderPlotsData, RenderPlots, PhoUIContainer

# %% [markdown]
# from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
# PyqtgraphRenderPlots, MatplotlibRenderPlots
# from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# %%

from pyphocorehelpers.DataStructure.general_parameter_containers import VisualizationParameters, RenderPlotsData, RenderPlots
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

# For scrollable BasicBinnedImageRenderingWindow
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import LayoutScrollability, _perform_build_root_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_ui, build_scrollable_graphics_layout_widget_with_nested_viewbox_ui


class BasicBinnedImageRenderingWindow(QtWidgets.QMainWindow):
    """ Renders a Matrix of binned data in the window.NonUniformImage and includes no histogram.
        NOTE: uses basic pg.ImageItem instead of pg.
        Observed to work well to display simple binned heatmaps/grids such as avg velocity across spatial bins, etc.    
        
        History:
            Based off of pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix.MatrixRenderingWindow
            
        Usage:
            from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
            out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_pf_2D_dt.xbin_labels, active_pf_2D_dt.ybin_labels, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity', scrollability_mode=LayoutScrollability.SCROLLABLE)
            out.add_data(row=1, col=0, matrix=active_eloy_analysis.pf_overlapDensity_2D, xbins=active_pf_2D_dt.xbin_labels, ybins=active_pf_2D_dt.ybin_labels, name='pf_overlapDensity', title='pf overlapDensity metric', variable_label='pf overlapDensity')
            out.add_data(row=2, col=0, matrix=active_pf_2D.ratemap.occupancy, xbins=active_pf_2D.xbin, ybins=active_pf_2D.ybin, name='occupancy_seconds', title='Seconds Occupancy', variable_label='seconds')
            out.add_data(row=3, col=0, matrix=active_simpler_pf_densities_analysis.n_neurons_meeting_firing_critiera_by_position_bins_2D, xbins=active_pf_2D.xbin, ybins=active_pf_2D.ybin, name='n_neurons_meeting_firing_critiera_by_position_bins_2D', title='# neurons > 1Hz per Pos (X, Y)', variable_label='# neurons')

    """
    
    def __init__(self, matrix=None, xbins=None, ybins=None, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity',
                 drop_below_threshold: float=0.0000001, color_map='viridis', color_bar_mode=None, wants_crosshairs=True, scrollability_mode=LayoutScrollability.SCROLLABLE, defer_show=False, **kwargs):
        super(BasicBinnedImageRenderingWindow, self).__init__(**kwargs)
        self.params = VisualizationParameters(name='BasicBinnedImageRenderingWindow')
        self.plots_data = RenderPlotsData(name='BasicBinnedImageRenderingWindow')
        self.plots = RenderPlots(name='BasicBinnedImageRenderingWindow')
        self.ui = PhoUIContainer(name='BasicBinnedImageRenderingWindow')
        self.ui.connections = PhoUIContainer(name='BasicBinnedImageRenderingWindow')

        self.params.scrollability_mode = LayoutScrollability.init(scrollability_mode)

        
        if isinstance(color_map, str):        
            self.params.colorMap = pg.colormap.get("viridis")
        else:
            # better be a ColorMap object directly
            assert isinstance(color_map, ColorMap)
            self.params.colorMap = color_map
            
        self.params.color_bar_mode = color_bar_mode
        if self.params.color_bar_mode == 'one':
            # Single shared color_bar between all items:
            self.params.shared_colorBarItem = pg.ColorBarItem(values=(0,1), colorMap=self.params.colorMap, label='all_pf_2Ds')
        else:
            self.params.shared_colorBarItem = None
            
        self.params.wants_crosshairs = wants_crosshairs

        pg.setConfigOption('imageAxisOrder', 'row-major') # Switch default order to Row-major

        ## Old (non-scrollable) way:        
        # self.ui.graphics_layout = pg.GraphicsLayoutWidget(show=True)
        # self.setCentralWidget(self.ui.graphics_layout)

        ## Build scrollable UI version:
        self.ui = _perform_build_root_graphics_layout_widget_ui(self.ui, is_scrollable=self.params.scrollability_mode.is_scrollable)
        if self.params.scrollability_mode.is_scrollable:
            self.setCentralWidget(self.ui.scrollAreaWidget)
        else:
            self.setCentralWidget(self.ui.graphics_layout)

        # Shared:
        self.setWindowTitle(title)
        self.resize(1000, 800)
        
        ## Add Label for debugging:
        self.ui.mainLabel = pg.LabelItem(justify='right')
        self.ui.graphics_layout.addItem(self.ui.mainLabel)
        
        # Add the item for the provided data:
        self.add_data(row=0, col=0, matrix=matrix, xbins=xbins, ybins=ybins, name=name, title=title, variable_label=variable_label, drop_below_threshold=drop_below_threshold)
        
        if not defer_show:
            self.show()

# %%
## Saving



# %%
def save_figure(self, export_path: Path):
	""" Exports all four rasters to a specified file path
	
	_out_rank_order_event_raster_debugger.save_figure(export_path=export_path)
	
	"""
	save_paths = []
	# root_plots_dict = {k:v['root_plot'] for k,v in _out_rank_order_event_raster_debugger.plots.all_separate_plots.items()} # PlotItem 

	root_plots_dict = self.root_plots_dict
	root_plots_dict['long_LR'].setYRange(-0.5, float(self.max_n_neurons))

	for a_decoder, a_plot in root_plots_dict.items():
		a_plot.setYRange(-0.5, float(self.max_n_neurons))
		self.get_epoch_active_aclus()
		out_path = export_path.joinpath(f'{a_decoder}_plot.png').resolve()
		export_pyqtgraph_plot(a_plot, savepath=out_path, background=pg.mkColor(0, 0, 0, 0))
		save_paths.append(out_path)

	return save_paths



# %% [markdown]
# ## Combine separate figures in pyqtgraph
# Exactly what we need to combine the separate figures except this uses pyqtgraph instead of matplotlib
# 

# %% [markdown]
# 
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:261)
# ```python
# # From `GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger.init_rank_order_debugger`
# #TODO 2023-11-30 15:14: - [ ] Unpacking and putting in docks and such not yet finished. Update functions would need to be done separately.
# rasters_display_outputs = _obj.plots.rasters_display_outputs
# all_apps = {a_decoder_name:a_raster_setup_tuple.app for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_windows = {a_decoder_name:a_raster_setup_tuple.win for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_plots = {a_decoder_name:a_raster_setup_tuple.plots for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_plots_data = {a_decoder_name:a_raster_setup_tuple.plots_data for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# 
# main_plot_identifiers_list = list(all_windows.keys()) # ['long_LR', 'long_RL', 'short_LR', 'short_RL']
# 
# ## Extract the data items:
# all_separate_data_all_spots = {a_decoder_name:a_raster_setup_tuple.plots_data.all_spots for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_data_all_scatterplot_tooltips_kwargs = {a_decoder_name:a_raster_setup_tuple.plots_data.all_scatterplot_tooltips_kwargs for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_data_new_sorted_rasters = {a_decoder_name:a_raster_setup_tuple.plots_data.new_sorted_raster for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_data_spikes_dfs = {a_decoder_name:a_raster_setup_tuple.plots_data.spikes_df for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# 
# # Extract the plot/renderable items
# all_separate_root_plots = {a_decoder_name:a_raster_setup_tuple.plots.root_plot for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_grids = {a_decoder_name:a_raster_setup_tuple.plots.grid for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_scatter_plots = {a_decoder_name:a_raster_setup_tuple.plots.scatter_plot for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# all_separate_debug_header_labels = {a_decoder_name:a_raster_setup_tuple.plots.debug_header_label for a_decoder_name, a_raster_setup_tuple in rasters_display_outputs.items()}
# 
# # Embedding in docks:
# root_dockAreaWindow, app = DockAreaWrapper.build_default_dockAreaWindow(title='Pho Debug Plot Directional Template Rasters')
# icon = try_get_icon(icon_path=":/Icons/Icons/visualizations/template_1D_debugger.ico")
# if icon is not None:
#     root_dockAreaWindow.setWindowIcon(icon)
# 
# ## Build Dock Widgets:
# def get_utility_dock_colors(orientation, is_dim):
#     """ used for CustomDockDisplayConfig for non-specialized utility docks """
#     # Common to all:
#     if is_dim:
#         fg_color = '#aaa' # Grey
#     else:
#         fg_color = '#fff' # White
# 
#     # a purplish-royal-blue
#     if is_dim:
#         bg_color = '#d8d8d8'
#         border_color = '#717171'
#     else:
#         bg_color = '#9d9d9d'
#         border_color = '#3a3a3a'
# 
#     return fg_color, bg_color, border_color
# 
# 
# # decoder_names_list = ('long_LR', 'long_RL', 'short_LR', 'short_RL')
# _out_dock_widgets = {}
# dock_configs = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False),
#                 CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_LR_dock_colors, showCloseButton=False), CustomDockDisplayConfig(custom_get_colors_callback_fn=DisplayColorsEnum.Laps.get_RL_dock_colors, showCloseButton=False))))
# # dock_add_locations = (['left'], ['left'], ['right'], ['right'])
# # dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['right'], ['right'], ['right'], ['right'])))
# dock_add_locations = dict(zip(('long_LR', 'long_RL', 'short_LR', 'short_RL'), (['left'], ['bottom'], ['right'], ['right'])))
# 
# for i, (a_decoder_name, a_win) in enumerate(all_windows.items()):
#     if (a_decoder_name == 'short_RL'):
#         short_LR_dock = root_dockAreaWindow.find_display_dock('short_LR')
#         assert short_LR_dock is not None
#         dock_add_locations['short_RL'] = ['bottom', short_LR_dock]
#         print(f'using overriden dock location.')
# 
#     _out_dock_widgets[a_decoder_name] = root_dockAreaWindow.add_display_dock(identifier=a_decoder_name, widget=a_win, dockSize=(300,600), dockAddLocationOpts=dock_add_locations[a_decoder_name], display_config=dock_configs[a_decoder_name], autoOrientation=False)
# 
# 
# # Build callback functions:
# def on_update_active_scatterplot_kwargs(override_scatter_plot_kwargs):
#     """ captures: main_plot_identifiers_list, plots, plots_data """
#     for _active_plot_identifier in main_plot_identifiers_list:
#         # for _active_plot_identifier, a_scatter_plot in plots.scatter_plots.items():
#         # new_ax = plots.ax[_active_plot_identifier]
#         a_scatter_plot = all_separate_scatter_plots[_active_plot_identifier]
#         plots_data = all_separate_plots_data[_active_plot_identifier]
#         a_scatter_plot.setData(plots_data.seperate_all_spots_dict[_active_plot_identifier], **(plots_data.seperate_all_scatterplot_tooltips_kwargs_dict[_active_plot_identifier] or {}), **override_scatter_plot_kwargs)
# 
# def on_update_active_epoch(an_epoch_idx, an_epoch):
#     """ captures: main_plot_identifiers_list, all_separate_root_plots """
#     for _active_plot_identifier in main_plot_identifiers_list:
#         new_ax = all_separate_root_plots[_active_plot_identifier]
#         print(f'an_epoch: {an_epoch}')
#         new_ax.setXRange(an_epoch.start, an_epoch.stop)
#         new_ax.setAutoPan(False)
#         # new_ax.getAxis('left').setLabel(f'[{an_epoch.label}]')
# 
#         # a_scatter_plot = plots.scatter_plots[_active_plot_identifier]
# 
# 
# ## Build the utility controls at the bottom:
# ctrls_dock_config = CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False)
# 
# ctrls_widget = ScrollBarWithSpinBox()
# ctrls_widget.setObjectName("ctrls_widget")
# ctrls_widget.update_range(0, (_obj.n_epochs-1))
# ctrls_widget.setValue(10)
# 
# def valueChanged(new_val:int):
#     print(f'valueChanged(new_val: {new_val})')
#     _obj.on_update_epoch_IDX(int(new_val))
# 
# ctrls_widget_connection = ctrls_widget.sigValueChanged.connect(valueChanged)
# ctrl_layout = pg.LayoutWidget()
# ctrl_layout.addWidget(ctrls_widget, row=1, rowspan=1, col=1, colspan=2)
# ctrl_widgets_dict = dict(ctrls_widget=ctrls_widget, ctrls_widget_connection=ctrls_widget_connection)
# 
# # Step 4: Create DataFrame and QTableView
# # df =  selected active_selected_spikes_df # pd.DataFrame(...)  # Replace with your DataFrame
# # model = PandasModel(df)
# # pandasDataFrameTableModel = SimplePandasModel(active_epochs_df.copy())
# 
# # tableView = pg.QtWidgets.QTableView()
# # tableView.setModel(pandasDataFrameTableModel)
# # tableView.setObjectName("pandasTablePreview")
# # # tableView.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)
# 
# # ctrl_widgets_dict['pandasDataFrameTableModel'] = pandasDataFrameTableModel
# # ctrl_widgets_dict['tableView'] = tableView
# 
# # # Step 5: Add TableView to LayoutWidget
# # ctrl_layout.addWidget(tableView, row=2, rowspan=1, col=1, colspan=1)
# 
# 
# # Tabbled table widget:
# tab_widget, views_dict, models_dict = create_tabbed_table_widget(dataframes_dict={'epochs': active_epochs_df.copy(),
#                                                                                                 'spikes': global_spikes_df.copy(), 
#                                                                                                 'combined_epoch_stats': pd.DataFrame()})
# ctrl_widgets_dict['tables_tab_widget'] = tab_widget
# ctrl_widgets_dict['views_dict'] = views_dict
# ctrl_widgets_dict['models_dict'] = models_dict
# 
# 
# 
# # Add the tab widget to the layout
# ctrl_layout.addWidget(tab_widget, row=2, rowspan=1, col=1, colspan=1)
# 
# 
# logTextEdit = pg.QtWidgets.QTextEdit()
# logTextEdit.setReadOnly(True)
# logTextEdit.setObjectName("logTextEdit")
# # logTextEdit.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)
# 
# ctrl_layout.addWidget(logTextEdit, row=2, rowspan=1, col=2, colspan=1)
# 
# _out_dock_widgets['bottom_controls'] = root_dockAreaWindow.add_display_dock(identifier='bottom_controls', widget=ctrl_layout, dockSize=(600,200), dockAddLocationOpts=['bottom'], display_config=ctrls_dock_config)
# 
# ## Add two labels in the top row that show the Long/Short column values:
# long_short_info_layout = pg.LayoutWidget()
# long_short_info_layout.setObjectName('layoutLongShortInfo')
# 
# long_info_label = long_short_info_layout.addLabel(text='LONG', row=0, col=0)
# long_info_label.setObjectName('lblLongInfo')
# # long_info_label.setAlignment(pg.QtCore.Qt.AlignCenter)
# long_info_label.setAlignment(pg.QtCore.Qt.AlignLeft)
# 
# short_info_label = long_short_info_layout.addLabel(text='SHORT', row=0, col=1)
# short_info_label.setObjectName('lblShortInfo')
# # short_info_label.setAlignment(pg.QtCore.Qt.AlignCenter)
# short_info_label.setAlignment(pg.QtCore.Qt.AlignRight)
# 
# _out_dock_widgets['LongShortColumnsInfo_dock'] = root_dockAreaWindow.add_display_dock(identifier='LongShortColumnsInfo_dock', widget=long_short_info_layout, dockSize=(600,60), dockAddLocationOpts=['top'], display_config=CustomDockDisplayConfig(custom_get_colors_callback_fn=get_utility_dock_colors, showCloseButton=False, corner_radius='0px'))
# _out_dock_widgets['LongShortColumnsInfo_dock'][1].hideTitleBar() # hide the dock title bar
# 
# # Add the widgets to the .ui:
# long_short_info_layout = long_short_info_layout
# long_info_label = long_info_label
# short_info_label = short_info_label
# info_labels_widgets_dict = dict(long_short_info_layout=long_short_info_layout, long_info_label=long_info_label, short_info_label=short_info_label)
# 
# root_dockAreaWindow.resize(600, 900)
# 
# ## Build final .plots and .plots_data:
# _obj.plots = RenderPlots(name=name, root_dockAreaWindow=root_dockAreaWindow, apps=all_apps, all_windows=all_windows, all_separate_plots=all_separate_plots,
#                             root_plots=all_separate_root_plots, grids=all_separate_grids, scatter_plots=all_separate_scatter_plots, debug_header_labels=all_separate_debug_header_labels,
#                             dock_widgets=_out_dock_widgets, text_items_dict=None) # , ctrl_widgets={'slider': slider}
# _obj.plots_data = RenderPlotsData(name=name, main_plot_identifiers_list=main_plot_identifiers_list,
#                                     seperate_all_spots_dict=all_separate_data_all_spots, seperate_all_scatterplot_tooltips_kwargs_dict=all_separate_data_all_scatterplot_tooltips_kwargs, seperate_new_sorted_rasters_dict=all_separate_data_new_sorted_rasters, seperate_spikes_dfs_dict=all_separate_data_spikes_dfs,
#                                     on_update_active_epoch=on_update_active_epoch, on_update_active_scatterplot_kwargs=on_update_active_scatterplot_kwargs, **{k:v for k, v in _obj.plots_data.to_dict().items() if k not in ['name']})
# _obj.ui = PhoUIContainer(name=name, app=app, root_dockAreaWindow=root_dockAreaWindow, ctrl_layout=ctrl_layout, **ctrl_widgets_dict, **info_labels_widgets_dict, on_valueChanged=valueChanged, logTextEdit=logTextEdit, dock_configs=dock_configs, controlled_references=None)
# _obj.params = VisualizationParameters(name=name, is_laps=False, enable_show_spearman=True, enable_show_pearson=False, enable_show_Z_values=True, use_plaintext_title=False, **param_kwargs)
# 
# 
# cls.try_build_selected_spikes(_obj)
# 
# _obj._build_cell_y_labels() # builds the cell labels
# 
# ## Cleanup when done:
# for a_decoder_name, a_root_plot in _obj.plots.root_plots.items():
#     a_root_plot.setTitle(title=a_decoder_name)
#     # a_root_plot.setTitle(title="")
#     a_left_axis = a_root_plot.getAxis('left')# axisItem
#     a_left_axis.setLabel(a_decoder_name)
#     a_left_axis.setStyle(showValues=False)
#     a_left_axis.setTicks([])
#     # a_root_plot.hideAxis('bottom')
#     # a_root_plot.hideAxis('bottom')
#     a_root_plot.hideAxis('left')
#     a_root_plot.setYRange(-0.5, float(_obj.max_n_neurons))
#     
# 
# # for a_decoder_name, a_scatter_plot_item in _obj.plots.scatter_plots.items():
# #     a_scatter_plot_item.hideAxis('left')
# 
# # Hide the debugging labels
# for a_decoder_name, a_label in _obj.plots.debug_header_labels.items():
#     # a_label.setText('NEW')
#     a_label.hide() # hide the labels unless we need them.
# 
# _obj.register_internal_callbacks()
# 
# 
# ctrl_widgets_dict['models_dict']['combined_epoch_stats'] = SimplePandasModel(_obj.combined_epoch_stats_df.copy())
# 
# # Create and associate view with model
# # view = pg.QtWidgets.QTableView()
# ctrl_widgets_dict['views_dict']['combined_epoch_stats'].setModel(ctrl_widgets_dict['models_dict']['combined_epoch_stats'])
# 
# 
# return _obj
# 
# 
# ```

# %% [markdown]
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:726](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/ContainerBased/RankOrderRastersDebugger.py:726)
# exports
# ```python
# # From `GUI.PyQtPlot.Widgets.ContainerBased.RankOrderRastersDebugger.export_figure_all_slider_values`
# @function_attributes(short_name=None, tags=['figure', 'debug'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-12-21 19:49', related_items=[])
# def export_figure_all_slider_values(self, export_path: Union[str,Path]):
#     """ sweeps the rank_order_event_raster_debugger through its various slider values, exporting all four of its plots as images for each value. 
# 
#     Usage:
#         export_path = Path(r'~/Desktop/2023-12-19 Exports').resolve()
#         all_save_paths = _out_rank_order_event_raster_debugger.export_figure_all_slider_values(export_path=export_path)
# 
# 
#     """
#     all_save_paths = {}
# 
#     for i in np.arange(0, self.n_epochs, 5):
#         self.ui.ctrls_widget.setValue(i) ## Adjust the slider, using its callbacks as well to update the displayed epoch.
#         
#         # _out_rank_order_event_raster_debugger.on_update_epoch_IDX(an_epoch_idx=i)
#         active_epoch_label = self.active_epoch_label
# 
#         save_paths = []
# 
#         for a_decoder, a_plot in self.root_plots_dict.items():
#             curr_filename_prefix = f'Epoch{active_epoch_label}_{a_decoder}'
#             # a_plot.setYRange(-0.5, float(self.max_n_neurons))
#             out_path = export_path.joinpath(f'{curr_filename_prefix}_plot.png').resolve()
#             export_pyqtgraph_plot(a_plot, savepath=out_path, background=pg.mkColor(0, 0, 0, 0))
#             save_paths.append(out_path)
# 
#         all_save_paths[active_epoch_label] = save_paths
#     
#     return all_save_paths
# ```

# %% [markdown]
# ## 📚 Save figures to disk/output figures
# 

# %%
complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()

curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='decoded_P_Short_Posterior'), fig=matplotlib_fig)



# %%
complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()

curr_active_pipeline.output_figure(final_context=complete_session_context.overwriting_context(display='pos_over_t'), fig=widget.getRootPlotItem())


# %% [markdown]
# ## 👨🏻‍💻📚 Computation Functions Documentation Guide

# %% [markdown]
# ### `curr_active_pipeline.perform_specific_computation(...)`: perform a specific computation (specified in computation_functions_name_includelist) in a minimally destructive manner using the previously recomputed results

# %%
curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['rank_order_shuffle_analysis','_add_extended_pf_peak_information',
 '_build_trial_by_trial_activity_metrics',
 '_decode_and_evaluate_epochs_using_directional_decoders',
 '_decode_continuous_using_directional_decoders',
 '_decoded_epochs_heuristic_scoring',
 '_split_train_test_laps_data',
 'perform_wcorr_shuffle_analysis'], computation_kwargs_list=[{'num_shuffles': 100, 'skip_laps': False, 'minimum_inclusion_fr_Hz':2.0, 'included_qclu_values':[1,2,4,5,6,7]}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)


# %% [markdown]
# ### `batch_evaluate_required_computations` - checks if computations with specified names are completed or still needed

# %%
# extended_computations_include_includelist=['ratemap_peaks_prominence2d', 'rank_order_shuffle_analysis', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring',] # do only specified
extended_computations_include_includelist=['rank_order_shuffle_analysis', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'ratemap_peaks_prominence2d', ] # do only specified
needs_computation_output_dict, valid_computed_results_output_list, remaining_include_function_names = batch_evaluate_required_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                    force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
print(f'Post-load global computations: needs_computation_output_dict: {[k for k,v in needs_computation_output_dict.items() if (v is not None)]}')

# %% [markdown]
# ### `batch_extended_computations` - I think this is the bad/inefficient version that fully replaces computation_results and loses any computations not specified

# %%
newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=False, progress_print=True,
                                                    force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
newly_computed_values

# %% [markdown]
# # Create new Computation Files

# %% [markdown]
# ### Registering new computation parameters

# %% [markdown]
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py:1391](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py:1391)
# ```python
#     lap_estimation_parameters = curr_active_pipeline.sess.config.preprocessing_parameters.epoch_estimation_parameters.laps
#     assert lap_estimation_parameters is not None
#     
#     use_direction_dependent_laps: bool = lap_estimation_parameters.get('use_direction_dependent_laps', True)
#     print(f'constrain_to_laps(...): use_direction_dependent_laps: {use_direction_dependent_laps}')
# ```
# 

# %%
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import MultiContextComputationFunctions

curr_active_pipeline.register_computation(computation_function=MultiContextComputationFunctions._perform_jonathan_replay_firing_rate_analyses, is_global=True, registered_name='_perform_jonathan_replay_firing_rate_analyses')

# %% [markdown]
# ### TODO 2023-08-21 17:36: - [ ] Dealing with Configurations:

# %%
""" #TODO 2023-08-21 17:36: - [ ] Dealing with Configurations:

# Universal Computation parameter specifications

- [ ] should be serializable to HDF
- [ ] should allow accessing properties by a hierarchical "grouping" structure

- [ ] ideally would associate parameters with the computations that use them (although this introduces coupling)

- [ ] should allow both "path" or "object-property (dot)" access
config['pf_params/speed_thresh'] == config['pf_params']['speed_thresh'] == config.pf_params.speed_thresh == config.pf_params['speed_thresh'] == config['pf_params'].speed_thresh

- [ ] should be able to hold Objects in addition to raw Python types (config.pf_params.computation_epochs = Epoch(...))

- [ ] must support type-hinting and ipython auto-completion

- [ ] ideally could be added from global function specification

@register_global_computation_parameter(..., instantaneous_time_bin_size_seconds: float = 0.01)
def _perform_long_short_instantaneous_spike_rate_groups_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
	print(global_computation_results.computation_config.instantaneous_time_bin_size_seconds)


computation_results: dict
│   ├── maze1: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult
    │   ├── sess: neuropy.core.session.dataSession.DataSession
    │   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── maze2: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult
    │   ├── sess: neuropy.core.session.dataSession.DataSession
    │   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── maze: pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult
    │   ├── sess: neuropy.core.session.dataSession.DataSession
    │   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    │   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
    
    
print_keys_if_possible("computation_results['maze'].computation_config", curr_active_pipeline.computation_results['maze'].computation_config, max_depth=3)

computation_results['maze'].computation_config: neuropy.utils.dynamic_container.DynamicContainer
│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
    │   ├── speed_thresh: float
    │   ├── grid_bin: tuple - (2,)
    │   ├── grid_bin_bounds: tuple - (2, 2)
    │   ├── smooth: tuple - (2,)
    │   ├── frate_thresh: float
    │   ├── time_bin_size: float
    │   ├── computation_epochs: neuropy.core.epoch.Epoch
        │   ├── _filename: NoneType
        │   ├── _metadata: NoneType
        │   ├── _df: pandas.core.frame.DataFrame (children omitted) - (80, 6)
│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
    │   ├── max_num_spikes_per_neuron: int
    │   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
        │   ├── s: int
        │   ├── gamma: float
    │   ├── use_progress_bar: bool
    │   ├── debug_print: bool
    
    
        
global_computation_results: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── sess: neuropy.core.session.dataSession.DataSession
│   ├── computation_config: NoneType
│   ├── computed_data: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── accumulated_errors: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
│   ├── computation_times: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters

    
"""



# %% [markdown]
# ### Properties:

# %%
curr_active_pipeline.global_computation_results

# %%
curr_active_pipeline.computation_results

# %% [markdown]
# #### Local Computation Functions

# %%
@function_attributes(short_name='firing_rate_trends', tags=[''],
                        input_requires=["computation_result.sess.spikes_df", "computation_result.computed_data['pf2D']"], output_provides=["computation_result.computed_data['firing_rate_trends']"],
                        uses=[], used_by=[], creation_date='2023-08-31 00:00', related_items=[],
                        validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['firing_rate_trends']['pf_included_spikes_only']), is_global=False)
def _perform_firing_rate_trends_computation(computation_result: ComputationResult, debug_print=False):

# %% [markdown]
# #### Global Computation Functions

# %%
owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False

# %% [markdown]
# ### Registering a new computation function

# %%
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.MultiContextComputationFunctions import MultiContextComputationFunctions

curr_active_pipeline.register_computation(computation_function=MultiContextComputationFunctions._perform_jonathan_replay_firing_rate_analyses, is_global=True, registered_name='_perform_jonathan_replay_firing_rate_analyses')

# %% [markdown]
# ### Performing a specific computation:

# %%
curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_perform_jonathan_replay_firing_rate_analyses'], fail_on_exception=True, debug_print=True) # , progress_logger_callback=print

# %% [markdown]
# #### You can provide `computation_kwargs_list=[{'time_bin_size': 0.02}]` arguments, a list of kwargs corresponding to each function in `computation_functions_name_includelist=['_decode_continuous_using_directional_decoders']`

# %%
curr_active_pipeline.reload_default_computation_functions()
curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['_decode_continuous_using_directional_decoders'], computation_kwargs_list=[{'time_bin_size': 0.02}], enabled_filter_names=None, fail_on_exception=True, debug_print=False)

# %%
curr_active_pipeline.save_pipeline()

# %%


# %% [markdown]
# ### Computation Validators:
# a computation validator is a function that is ran to determine whether a computation function needs to be executed (or whether the previous results can be used).
# An instance of `pyphoplacecellanalysis.General.Model.SpecificComputationValidation.SpecificComputationValidator`
# 
# #### **WARNING!** **IMPORTANT** Pickling Validators - Because they're used in @function_attributes, the `validate_compputation_test` is attempted to be pickled (which is undesirable). Since only top-level definitions can be pickled this creates an issue where you can't use `SomeClass.perform_validate_computation` as a validator. Instead a separate top-level wrapper must be made.
# #TODO 2024-03-13 18:27: - [ ] Remove the `validate_compputation_test` property from the pickle dictionary of `function_attributes`.
# 

# %% [markdown]
# Often defined ineline like `validate_computation_test: `
# ```python
# @function_attributes(short_name='split_to_directional_laps', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D'], input_requires=[], output_provides=[], uses=['_perform_PBE_stats'], used_by=[], creation_date='2023-10-25 09:33', related_items=[],
#         validate_computation_test=DirectionalLapsHelpers.validate_has_directional_laps, is_global=True)
#     def _split_to_directional_laps(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False):
#         # implementation here
# ```

# %% [markdown]
# #### 2024-01-06 - SpecificComputationValidation

# %%
from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import SpecificComputationResultsSpecification
from pyphoplacecellanalysis.General.Model.SpecificComputationValidation import SpecificComputationValidator
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_extended_computations

curr_active_pipeline.reload_default_computation_functions()

_test_extended_computations_include_includelist = ['pf_computation', # 'pfdt_computation', 'firing_rate_trends',
    # 'pf_dt_sequential_surprise',
    # 'extended_stats',
    # 'long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', # 'long_short_rate_remapping',
    # 'ratemap_peaks_prominence2d',
    # 'long_short_inst_spike_rate_groups',
    # 'long_short_endcap_analysis',
    # 'spike_burst_detection',
    'split_to_directional_laps',
    'merged_directional_placefields',
    'rank_order_shuffle_analysis',
]

_test_force_recompute_override_computations_includelist = ['rank_order_shuffle_analysis']


# # batch_extended_computations
batch_extended_computations(curr_active_pipeline, include_includelist=_test_extended_computations_include_includelist, include_global_functions=True, fail_on_exception=True,
	 force_recompute=False, force_recompute_override_computations_includelist=_test_force_recompute_override_computations_includelist,
	 dry_run=False)


# %% [markdown]
# ### - [ ] Step 2 - Add the `@function_attributes(short_name='directional_decoders_epoch_heuristic_scoring',` to the appropriate list in `NonInteractiveProcessing`
# 
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py:312](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py:312)
# ```python
# # From `General.Batch.NonInteractiveProcessing.non_global_comp_names`
# non_global_comp_names = ['lap_direction_determination', 'pf_computation', 'pfdt_computation', 'firing_rate_trends', 'pf_dt_sequential_surprise', 'ratemap_peaks_prominence2d', 'position_decoding', 'position_decoding_two_step', 'spike_burst_detection']
# global_comp_names = ['long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', 'long_short_rate_remapping', 'long_short_inst_spike_rate_groups', 'pf_dt_sequential_surprise', 'long_short_endcap_analysis',
#                         'split_to_directional_laps', 'merged_directional_placefields', 'rank_order_shuffle_analysis', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring', '***YOUR_NEW_COMP_FN_SHORT_NAME***']
# 
# ```

# %%

dry_run = True
include_includelist=None #_test_extended_computations_include_includelist
include_global_functions=True
force_recompute=False
force_recompute_override_computations_includelist=_test_force_recompute_override_computations_includelist
included_computation_filter_names = None

non_global_comp_names = ['pf_computation', 'pfdt_computation', 'firing_rate_trends', 'pf_dt_sequential_surprise', 'ratemap_peaks_prominence2d', 'position_decoding', 'position_decoding_two_step', 'spike_burst_detection']
global_comp_names = ['long_short_decoding_analyses', 'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', 'long_short_rate_remapping', 'long_short_inst_spike_rate_groups', 'pf_dt_sequential_surprise', 'long_short_endcap_analysis',
						'split_to_directional_laps', 'merged_directional_placefields', 'rank_order_shuffle_analysis'] # , 'long_short_rate_remapping'

if include_includelist is None:
	# include all:
	include_includelist = non_global_comp_names + global_comp_names
else:
	print(f'included includelist is specified: {include_includelist}, so only performing these extended computations.')


_, _, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
if included_computation_filter_names is None:
	included_computation_filter_names = [global_epoch_name] # use only the global epoch: e.g. ['maze']


## Hardcoded comp_specifiers
_comp_specifiers = list(curr_active_pipeline.get_merged_computation_function_validators().values())
## Execution order is currently determined by `_comp_specifiers` order and not the order the `include_includelist` lists them (which is good) but the `curr_active_pipeline.registered_merged_computation_function_dict` has them registered in *REVERSE* order for the specific computation function called, so we need to reverse these
_comp_specifiers = reversed(_comp_specifiers)

for _comp_specifier in _comp_specifiers:
	if (not _comp_specifier.is_global) or include_global_functions:
		if (_comp_specifier.short_name in include_includelist) or (_comp_specifier.computation_fn_name in include_includelist):
			if (not _comp_specifier.is_global):
				# Not Global-only, need to compute for all `included_computation_filter_names`:
				for a_computation_filter_name in included_computation_filter_names:
					if not dry_run:
						newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, computation_filter_name=a_computation_filter_name, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=force_recompute)
					else:
						print(f'dry-run: {_comp_specifier.short_name}, computation_filter_name={a_computation_filter_name}, force_recompute={force_recompute}')

			else:
				# Global-Only:
				_curr_force_recompute = force_recompute or ((_comp_specifier.short_name in force_recompute_override_computations_includelist) or (_comp_specifier.computation_fn_name in force_recompute_override_computations_includelist)) # force_recompute for this specific result if either of its name is included in `force_recompute_override_computations_includelist`
				if not dry_run:
					newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, computation_filter_name=global_epoch_name, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=_curr_force_recompute)
				else:
					print(f'dry-run: {_comp_specifier.short_name}, force_recompute={force_recompute}, curr_force_recompute={_curr_force_recompute}')
					# _comp_specifier.debug_comp_validator_status(curr_active_pipeline.global_computation_results)
					# Check for existing result:
					is_known_missing_provided_keys: bool = _comp_specifier.try_check_missing_provided_keys(curr_active_pipeline.global_computation_results)
					if is_known_missing_provided_keys:
						print(f'\t{_comp_specifier.short_name} -- is_known_missing_provided_keys = True!')




	# missing_keys_dict

# try_computation_if_needed

# %% [markdown]
# From an @function_attributes(..., validation_computation_test=...) definition, the validator is built using: `SpecificComputationValidator.init_from_decorated_fn(...)``
# 
# 

# %%


# %% [markdown]
# Computation is performed by calling `.try_computation_if_needed(...)`

# %% [markdown]
# ### Specifying Data Dependencies/Provides for Computation and Display Functions
# 
# #### `input_requires` and `output_provides`:
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/DefaultComputationFunctions.py:81](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/DefaultComputationFunctions.py:81)
# ```python
#     @function_attributes(short_name='position_decoding_two_step', tags=['decoding', 'position', 'two-step'],
#                           input_requires=["computation_result.computed_data['pf1D_Decoder']", "computation_result.computed_data['pf2D_Decoder']"], output_provides=["computation_result.computed_data['pf1D_TwoStepDecoder']", "computation_result.computed_data['pf2D_TwoStepDecoder']"],
#                           uses=[], used_by=[], creation_date='2023-09-12 17:32', related_items=[],
#         validate_computation_test=lambda curr_active_pipeline, computation_filter_name='maze': (curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf1D_TwoStepDecoder'], curr_active_pipeline.computation_results[computation_filter_name].computed_data['pf2D_TwoStepDecoder']), is_global=False)
#     def _perform_two_step_position_decoding_computation(computation_result: ComputationResult, debug_print=False, **kwargs):
#         """ Builds the Zhang Velocity/Position For 2-step Bayesian Decoder for 2D Placefields
#         """\
#         ...
# ```
# 
# 
# #### `requires_global_keys` and `provides_global_keys`:
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py:6454](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py:6454)
# ```python
# # From `General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions._decode_continuous_using_directional_decoders`
#     @function_attributes(short_name='directional_decoders_decode_continuous', tags=['directional_pf', 'laps', 'epoch', 'session', 'pf1D', 'pf2D', 'continuous'], input_requires=[], output_provides=[], uses=['DirectionalDecodersContinuouslyDecodedResult'], used_by=[], creation_date='2024-01-17 09:05', related_items=[],
#         requires_global_keys=['DirectionalLaps', 'DirectionalMergedDecoders'], provides_global_keys=['DirectionalDecodersDecoded'],
#         # validate_computation_test=DirectionalDecodersContinuouslyDecodedResult.validate_has_directional_decoded_continuous_epochs,
#         validate_computation_test=_workaround_validate_has_directional_decoded_continuous_epochs,
#         is_global=True, computation_precidence=(1002.0))
#     def _decode_continuous_using_directional_decoders(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist=None, debug_print=False, time_bin_size: Optional[float]=None, should_disable_cache: bool = False):
#         """ Using the four 1D decoders, decodes continously streams of positions from the neural activity for each.
#         
# ```
# 
# 

# %% [markdown]
# ### Dropping a specific computed result:

# %%
global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop = ['DirectionalDecodersDecoded'], debug_print=True)


# %% [markdown]
# ## Computation Classes Documentation

# %% [markdown]
# ## Time-Dependent Placefields Documentation:
# 
# ### Resetting State:
# `reset(self)`: """ used to reset the calculations to an initial value. """
#     `setup_time_varying(self)`: """ Initialize for the 0th timestamp """
# 
# ### Making Snapshots:
# `snapshot(self)`: """ takes a snapshot of the current values at this time."""    
#         
# ### Restore Snapshots:
# `restore_from_snapshot(self, snapshot_t)`
#     `apply_snapshot_data(self, snapshot_t, snapshot_data)`
#   

# %%
# Reset the rebuild_fragile_linear_neuron_IDXs:
self._filtered_spikes_df, _reverse_cellID_index_map = self._filtered_spikes_df.spikes.rebuild_fragile_linear_neuron_IDXs()
self.fragile_linear_neuron_IDXs = np.unique(self._filtered_spikes_df.fragile_linear_neuron_IDX) # array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63])
self.n_fragile_linear_neuron_IDXs = len(self.fragile_linear_neuron_IDXs)
self._included_thresh_neurons_indx = np.arange(self.n_fragile_linear_neuron_IDXs)
self._peak_frate_filter_function = lambda list_: [list_[_] for _ in self._included_thresh_neurons_indx] # filter_function: takes any list of length n_neurons (original number of neurons) and returns only the elements that met the firing rate criteria
# ...
self.setup_time_varying()

# %%
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

# %% [markdown]
# # Aggregated Spike Information

# %%
`SpikeRateTrends`


# %%
time_binned_instantaneous_unit_specific_spike_rate = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.time_binned_instantaneous_unit_specific_spike_rate
timestamps = time_binned_instantaneous_unit_specific_spike_rate.time_bins

value_df = time_binned_instantaneous_unit_specific_spike_rate.instantaneous_unit_specific_spike_rate_values
value_df

# %%
# Number of spikes version:
time_binned_unit_specific_spike_rate = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis.time_binned_unit_specific_spike_rate
timestamps = time_binned_unit_specific_spike_rate.time_bins
value_df = time_binned_unit_specific_spike_rate.time_binned_unit_specific_binned_spike_rate
value_df

# %% [markdown]
# ## Data Structure Documentation Generation
# The functions below generate documentation in .md and .html format from passed data structures.

# %%
print_keys_if_possible('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'], custom_item_formatter=_rich_text_format_curr_value)

# %%
from ansi2html import Ansi2HTMLConverter # used by DocumentationFilePrinter to build html document from ansi-color coded version
from pyphocorehelpers.print_helpers import DocumentationFilePrinter

doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])

# %%
from ansi2html import Ansi2HTMLConverter # used by DocumentationFilePrinter to build html document from ansi-color coded version
from pyphocorehelpers.print_helpers import DocumentationFilePrinter

doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='InteractivePlaceCellConfig')
doc_printer.save_documentation('InteractivePlaceCellConfig', curr_active_pipeline.active_configs['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map', 'pf_listed_colormap'])
# doc_printer.reveal_output_files_in_system_file_manager()

# %%
doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='NeuropyPipeline')
doc_printer.save_documentation('NeuropyPipeline', curr_active_pipeline, non_expanded_item_keys=['_reverse_cellID_index_map', 'pf_listed_colormap', 'computation_results', 'active_configs', 'logger']) # 'Logger'

# %%
doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='DisplayPipelineStage')
doc_printer.save_documentation('DisplayPipelineStage', curr_active_pipeline.stage, non_expanded_item_keys=['_reverse_cellID_index_map', 'pf_listed_colormap', 'computation_results', 'active_configs', 'logger']) # 'Logger'

# %%
stage# doc_printer.reveal_output_files_in_system_file_manager()

# %%
filtered_context = curr_active_pipeline.filtered_contexts['maze1']
filtered_context.adding_context(collision_prefix='computation_params', comp_params_name=a_computation_config_name)

# %% [markdown]
# # PyQt Techniquies

# %% [markdown]
# ## PyQt Technique `@QtCore.Property(...)`

# %%
    @QtCore.Property(int) # Note that this ia *pyqt*Property, meaning it's available to pyqt
    def scheduledAnimationSteps(self):
        """The scheduledAnimationSteps property."""
        return self._scheduledAnimationSteps
    @scheduledAnimationSteps.setter
    def scheduledAnimationSteps(self, value):
        if self._scheduledAnimationSteps != value:
            # Only update if the value has changed from the previous one:
            self._scheduledAnimationSteps = value
            # TODO: maybe use a rate-limited signal that's emitted instead so this isn't called too often during interpolation?
            # self.shift_animation_frame_val(self._scheduledAnimationSteps) # TODO: this isn't quite right

# %% [markdown]
# ## PyQt Technique `pyqtExceptionPrintingSlot`:
# Note that this helps somewhat (prevents it from failing silently somewhat) but introduces other errors when exceptions occur.
# `from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot`

# %% [markdown]
# #### Before

# %%
...
	@QtCore.pyqtSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        # print(f'LiveWindowedData.on_window_duration_changed(start_t: {start_t}, end_t: {end_t}, duration: {duration})')
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.windowed_data_window_duration_changed_signal.emit(start_t, end_t, duration, data_value)
        
    @QtCore.pyqtSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        # if self.enable_debug_print:
        #     print(f'LiveWindowedData.on_window_changed(start_t: {start_t}, end_t: {end_t})')
        
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.windowed_data_window_updated_signal.emit(start_t, end_t, data_value)
        
    ## Called to update its internal TimeWindow
    @QtCore.pyqtSlot(float)
    def update_window_start(self, new_value):
        self.timeWindow.update_window_start(new_value)

    @QtCore.pyqtSlot(float, float)
    def update_window_start_end(self, new_start, new_end):
        self.timeWindow.update_window_start_end(new_start, new_end)
        
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @QtCore.pyqtSlot(object)
    def update_window_start_rate_limited(self, evt):
        self.update_window_start(*evt)
        
    @QtCore.pyqtSlot(object)
    def update_window_start_end_rate_limited(self, evt):
        self.update_window_start_end(*evt)

# %% [markdown]
# #### After `pyqtExceptionPrintingSlot` conversion:

# %%
from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot

...
	@pyqtExceptionPrintingSlot(float, float, float)
    def on_window_duration_changed(self, start_t, end_t, duration):
        """ changes self.half_render_window_duration """
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.windowed_data_window_duration_changed_signal.emit(start_t, end_t, duration, data_value)
        
    @pyqtExceptionPrintingSlot(float, float)
    def on_window_changed(self, start_t, end_t):
        # called when the window is updated
        # Get the data value from the internal data source
        data_value = self.dataSource.get_updated_data_window(start_t, end_t) # can return any value so long as it's an object
        self.windowed_data_window_updated_signal.emit(start_t, end_t, data_value)
        
    ## Called to update its internal TimeWindow
    @pyqtExceptionPrintingSlot(float)
    def update_window_start(self, new_value):
        self.timeWindow.update_window_start(new_value)

    @pyqtExceptionPrintingSlot(float, float)
    def update_window_start_end(self, new_start, new_end):
        self.timeWindow.update_window_start_end(new_start, new_end)
        
    ############### Rate-Limited SLots ###############:
    ##################################################
    ## For use with pg.SignalProxy
    # using signal proxy turns original arguments into a tuple
    @pyqtExceptionPrintingSlot(object)
    def update_window_start_rate_limited(self, evt):
        self.update_window_start(*evt)
        
    @pyqtExceptionPrintingSlot(object)
    def update_window_start_end_rate_limited(self, evt):
        self.update_window_start_end(*evt)


# %% [markdown]
# ## PyQt Technique `.installEventFilter(self)`:
# 
# `from pyphocorehelpers.gui.Qt.ExceptionPrintingSlot import pyqtExceptionPrintingSlot`

# %%
class EventListener(QWidget):
	def initUI(self):
        # ...
		# Connect signals to handle focus and editing states
        self.ui.jumpToHourMinSecTimeEdit.editingFinished.connect(self.on_jump_time_editing_finished)
        self.ui.jumpToHourMinSecTimeEdit.installEventFilter(self)


    def eventFilter(self, source, event):
        """Handle focus events to change styles dynamically."""
        if source == self.ui.jumpToHourMinSecTimeEdit:
            if event.type() == event.FocusIn:
                self.set_jump_time_white_style()
            elif event.type() == event.FocusOut:
                self.set_jump_time_light_grey_style()
                
        elif source == self.time_edit and event.type() == event.KeyPress:
            # """Handle Enter key to finalize and lose focus."""
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.time_edit.clearFocus()  # Finalize and lose focus
                return True  # Mark event as handled
            
        return super().eventFilter(source, event)


# %% [markdown]
# ## PyQt Technique `QTimer.singleShot(0, _subfn_META_RUN_ALL_BACKGROUND_FCNS)`:
# This **defers your callback until after the current batch of paint events completes**.
# Needed because pyqt rendering happens through the asynchronous event loop and paint events, and there’s no global “paint done” notification for an entire widget hierarchy.

# %%
class EventListener(QWidget):
	def initUI(self):
        # ...
		# Connect signals to handle focus and editing states
        self.ui.jumpToHourMinSecTimeEdit.editingFinished.connect(self.on_jump_time_editing_finished)
        self.ui.jumpToHourMinSecTimeEdit.installEventFilter(self)


    def eventFilter(self, source, event):
        """Handle focus events to change styles dynamically."""
        if source == self.ui.jumpToHourMinSecTimeEdit:
            if event.type() == event.FocusIn:
                self.set_jump_time_white_style()
            elif event.type() == event.FocusOut:
                self.set_jump_time_light_grey_style()
                
        elif source == self.time_edit and event.type() == event.KeyPress:
            # """Handle Enter key to finalize and lose focus."""
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.time_edit.clearFocus()  # Finalize and lose focus
                return True  # Mark event as handled
            
        return super().eventFilter(source, event)


# %% [markdown]
# # Planning 2023-11-21 - Easily Corresponding computation/display functions:

# %%
@define(slots=False)
class TheClass:
	a_prop1 = field()
	a_prop2 = field()


class TheClassDisplayFunctions:
	@display_function_for(TheClass)
	def a_display_fn_1(a_class_obj_inst: TheClass, *args, **kwargs):
		raise NotImplementedError


# %%
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\old_global_computation_results.pkl
W:\Data\Kdiba\vvp01\two\2006-4-10_12-58-3\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\vvp01\one\2006-4-10_12-25-50\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\vvp01\one\2006-4-09_17-29-30\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\gor01\two\2006-6-09_22-24-40\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\gor01\one\2006-6-09_1-22-43\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\gor01\one\2006-6-08_14-26-15\loadedSessPickle_2023-10-06.pkl
W:\Data\Kdiba\vvp01\two\2006-4-10_12-58-3\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\vvp01\two\2006-4-09_16-40-54\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\vvp01\one\2006-4-10_12-25-50\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\vvp01\one\2006-4-09_17-29-30\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-12_16-53-46\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-09_22-24-40\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-08_21-16-25\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\one\2006-6-09_1-22-43\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\one\2006-6-08_14-26-15\loadedSessPickle_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\global_computation_results_2023-10-05.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\output\global_computation_results.pkl
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\backup-20231113092010-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\backup-20231110234635-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\one\2006-6-09_1-22-43\backup-20231120234515-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\two\2006-4-09_16-40-54\backup-20231018001251-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\one\2006-4-10_12-25-50\backup-20231018000401-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\two\2006-4-10_12-58-3\backup-20231018002053-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\one\2006-4-09_17-29-30\backup-20231017235450-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-12_16-53-46\backup-20231017234659-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-09_22-24-40\backup-20231017234040-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\one\2006-6-12_15-55-31\backup-20231017231327-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-08_21-16-25\backup-20231017232341-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\two\2006-4-10_12-58-3\backup-20231006173854-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\two\2006-4-09_16-40-54\backup-20231006173714-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\one\2006-4-10_12-25-50\backup-20231006173124-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-08_21-16-25\backup-20231006171824-loadedSessPickle.pkl.bak
W:\Data\Kdiba\vvp01\one\2006-4-09_17-29-30\backup-20231006172839-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-09_22-24-40\backup-20231006172722-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\two\2006-6-12_16-53-46\backup-20231006172455-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\one\2006-6-12_15-55-31\backup-20231006165832-loadedSessPickle.pkl.bak
W:\Data\Kdiba\gor01\one\2006-6-09_1-22-43\2006-6-09_1-22-43.spk.6
W:\Data\Kdiba\gor01\two\2006-6-08_21-16-25\2006-6-08_21-16-25.spk.6
W:\Data\Kdiba\gor01\two\2006-6-08_21-16-25\2006-6-08_21-16-25.spk.11
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\2006-6-07_16-40-19.eeg
W:\Data\Kdiba\gor01\two\2006-6-07_16-40-19\20231113095949-loadedSessPickle.pkl


# %% [markdown]
# # PyQtGraph Reference Documentation

# %% [markdown]
# ### Changing Axis Tick-labels
# https://stackoverflow.com/a/32008832/9732163

# %%

left_axis = main_plot3.getAxis('left') # axisItem
tick_ydict = {y_pos:f"{int(aclu)}" for y_pos, aclu in zip(series_identity_y_values, sorted_neuron_ids)} # {0.5: '68', 1.5: '75', 2.5: '54', 3.5: '10', 4.5: '104', 5.5: '90', 6.5: '44', 7.5: '15', 8.5: '93', 9.5: '79', 10.5: '56', 11.5: '84', 12.5: '78', 13.5: '31', 14.5: '16', 15.5: '40', 16.5: '25', 17.5: '81', 18.5: '70', 19.5: '66', 20.5: '24', 21.5: '98', 22.5: '80', 23.5: '77', 24.5: '60', 25.5: '39', 26.5: '9', 27.5: '82', 28.5: '85', 29.5: '101', 30.5: '87', 31.5: '26', 32.5: '43', 33.5: '65', 34.5: '48', 35.5: '52', 36.5: '92', 37.5: '11', 38.5: '51', 39.5: '72', 40.5: '18', 41.5: '53', 42.5: '47', 43.5: '89', 44.5: '102', 45.5: '61'}
left_axis.setTicks([tick_ydict.items()])

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# # `matplotlib` Figure Best-practices

# %%
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
    
    perform_update_title_subtitle(fig=fig_long_pf_1D, ax=ax_long_pf_1D, title_string="TEST - 1D Placemaps", subtitle_string="TEST - SUBTITLE")

# %% [markdown]
# # 🌀🤷‍♀️🔀 Unsorted
# 

# %% [markdown]
# ## Robustly Accessing Computation Results (2024-08-30)
# These snippets can be used in a notebook to safely load a computation output (if it is present) or force compute it if it is missing.

# %% [markdown]
# ### Compute if missing (require result):
# ```python
# from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult
# 
# directional_trial_by_trial_activity_result: TrialByTrialActivityResult = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None)
# if directional_trial_by_trial_activity_result is None:
#     # if `KeyError: 'TrialByTrialActivity'` recompute
#     print(f'TrialByTrialActivity is not computed, computing it...')
#     curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['trial_by_trial_metrics'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
#     directional_trial_by_trial_activity_result = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None) ## try again to get the result
#     assert directional_trial_by_trial_activity_result is not None, f"directional_trial_by_trial_activity_result is None even after forcing recomputation!!"
#     print(f'\t done.')
# 
# ## unpack either way:
# any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
# active_pf_dt: PfND_TimeDependent = directional_trial_by_trial_activity_result.active_pf_dt
# directional_lap_epochs_dict: Dict[str, Epoch] = directional_trial_by_trial_activity_result.directional_lap_epochs_dict
# directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts
# ## OUTPUTS: directional_trial_by_trial_activity_result, directional_active_lap_pf_results_dicts
# 
# ```
# 
# 
# 
# ### Act if computed (optional action):
# ```python
# from pyphoplacecellanalysis.Analysis.reliability import TrialByTrialActivity
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrialByTrialActivityResult
# 
# directional_trial_by_trial_activity_result: TrialByTrialActivityResult = curr_active_pipeline.global_computation_results.computed_data.get('TrialByTrialActivity', None)
# 
# if directional_trial_by_trial_activity_result is not None:
#     any_decoder_neuron_IDs = directional_trial_by_trial_activity_result.any_decoder_neuron_IDs
#     active_pf_dt: PfND_TimeDependent = directional_trial_by_trial_activity_result.active_pf_dt
#     directional_lap_epochs_dict: Dict[str, Epoch] = directional_trial_by_trial_activity_result.directional_lap_epochs_dict
#     directional_active_lap_pf_results_dicts: Dict[str, TrialByTrialActivity] = directional_trial_by_trial_activity_result.directional_active_lap_pf_results_dicts
#     ## OUTPUTS: directional_trial_by_trial_activity_result, directional_active_lap_pf_results_dicts
# else:
#     print(f'TrialByTrialActivity is not computed.')
# ```

# %% [markdown]
# ## Computing pipeline with specific parameters with `override_parameters_flat_keypaths_dict` (2025-07-09)
# 
# 
# ```python
# override_parameters_flat_keypaths_dict = {'rank_order_shuffle_analysis.included_qclu_values': [1, 2, 4, 6, 7, 8, 9], 'rank_order_shuffle_analysis.minimum_inclusion_fr_Hz': 2.0} # e.g. {'rank_order_shuffle_analysis.included_qclu_values': [1, 2], 'rank_order_shuffle_analysis.minimum_inclusion_fr_Hz': 5.0,}
# ```
# 
# 
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py:93](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py:93)
# ```python
# from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder # for batch_load_session
# from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline, PipelineSavingScheme # for batch_load_session
# 
# # From `General.Batch.NonInteractiveProcessing.known_data_session_type_properties_dict`
# known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
# active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
# 
# active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
# active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]
# 
# ## Begin main run of the pipeline (load or execute):
# curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties,
#     override_basepath=Path(basedir), force_reload=force_reload, active_pickle_filename=active_pickle_filename, skip_save_on_initial_load=True, override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict)
# 
# curr_active_pipeline.update_parameters(override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict) # should already be updated, but try it again anyway.
# 
# was_loaded_from_file: bool =  curr_active_pipeline.has_associated_pickle # True if pipeline was loaded from an existing file, False if it was created fresh
# 	
# ```
# 
# 
# 

# %% [markdown]
# # Epoch Dataframe Metadata Access/Updating and Use
# `a_df.attrs.metadata = {}`

# %% [markdown]
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_completion_helpers.py:79](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_completion_helpers.py:79)
# ```python
# # From `General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_completion_helpers.replace_replay_epochs`
# 		## Set new:
# 		replay_estimation_parameters.epochs_source = new_replay_epochs.metadata.get('epochs_source', None)
# 		# replay_estimation_parameters.require_intersecting_epoch = None # don't actually purge these as I don't know what they are used for
# 		replay_estimation_parameters.min_inclusion_fr_active_thresh = new_replay_epochs.metadata.get('minimum_inclusion_fr_Hz', 1.0)
# 		replay_estimation_parameters.min_num_unique_aclu_inclusions = new_replay_epochs.metadata.get('min_num_active_neurons', 5)
# 
# ```
# 

# %% [markdown]
# ### Access in `PhoDibaPaper2024.ipynb.ipynb` to get figure context
# ```python
# from functools import partial
# from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import add_copy_save_action_buttons
# from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import plotly_pre_post_delta_scatter
# from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import _perform_plot_pre_post_delta_scatter
# from neuropy.utils.result_context import DisplaySpecifyingIdentifyingContext
# from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_helpers import PlotlyFigureContainer
# 
# ## INPUTS: earliest_delta_aligned_t_start, latest_delta_aligned_t_end
# is_dark_mode, template = PlotlyHelpers.get_plotly_template(is_dark_mode=False)
# # should_save: bool = True
# should_save: bool = False
# 
# 
# _perform_plot_pre_post_delta_scatter = partial(
#     _perform_plot_pre_post_delta_scatter,
#     time_delta_tuple=(earliest_delta_aligned_t_start, 0.0, latest_delta_aligned_t_end),
#     fig_size_kwargs=fig_size_kwargs,
#     is_dark_mode=is_dark_mode,
#     save_plotly=save_plotly,
# )
# 
# _perform_plot_pre_post_delta_scatter_with_embedded_context = partial(
#     _perform_plot_pre_post_delta_scatter,
#     data_context=None,
# )
# 
# ## Set dataframe context metadata
# 
# def _perform_update_df_context_metadata(data_context: IdentifyingContext, concatenated_ripple_df: pd.DataFrame):
#     """ sets the metadata in-place for the dataframe """
#     if concatenated_ripple_df is None:
#         print(f'WARN: dataframe passed is None! Attempted context: {data_context}. Skipping.')
#         return
#     concatenated_ripple_df.attrs.update(**dict(data_context=deepcopy(data_context)))
#     
# 
# _perform_update_df_context_metadata(data_context=IdentifyingContext(epochs_name='laps', data_grain='per_epoch', title_prefix="Lap Per Epoch", dataframe_name='df'), concatenated_ripple_df=all_sessions_laps_df)
# _perform_update_df_context_metadata(data_context=IdentifyingContext(epochs_name='laps', data_grain='per_time_bin', title_prefix="Lap Individual Time Bins", dataframe_name='time_bin_df'), concatenated_ripple_df=all_sessions_laps_time_bin_df)
# _perform_update_df_context_metadata(data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_epoch', title_prefix="PBE Per Epoch", dataframe_name='df'), concatenated_ripple_df = all_sessions_ripple_df)
# _perform_update_df_context_metadata(data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_time_bin', title_prefix="PBE Individual Time Bins", dataframe_name='time_bin_df'), concatenated_ripple_df = all_sessions_ripple_time_bin_df)
# _perform_update_df_context_metadata(data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_epoch', dataframe_name='MultiMeasure_ripple_df', title_prefix="multiMeasure - PBE Per Epoch"), concatenated_ripple_df = all_sessions_MultiMeasure_ripple_df)
# _perform_update_df_context_metadata(data_context = IdentifyingContext(epochs_name='PBE', data_grain='per_epoch', dataframe_name='all_scores_ripple_df', title_prefix="allScores - PBE Per Epoch"), concatenated_ripple_df = all_sessions_all_scores_ripple_df)
# _perform_update_df_context_metadata(data_context = IdentifyingContext(epochs_name='laps', data_grain='per_epoch', dataframe_name='MultiMeasure_laps_df', title_prefix="multiMeasure - Lap Per Epoch"), concatenated_ripple_df = all_sessions_MultiMeasure_laps_df)
# ```
# 

# %%


# %% [markdown]
# # Old Swap

# %%
a_meas_pos_line, _meas_pos_out_markers = DecodedTrajectoryMatplotlibPlotter._perform_plot_measured_position_line_helper(an_ax, a_measured_pos_df, a_time_bin_centers, fake_y_lower_bound=None, fake_y_upper_bound=None, rotate_to_vertical=True, debug_print=True)

# %% [markdown]
# # NDArray with Size Dimensions
# 
# I ran into some strange requirements for proper variable names in the Shape expression that I couldn't find stated in the documentation. Through trial and error I determined the following requirements:
# ## Rules:
# 1. You must always have an element type (even `Any`) -- `arr: NDArray[ND.Shape["N_ACLUS, N_TIME_BINS"], Any]` is valid, `arr: NDArray[ND.Shape["N_ACLUS, N_TIME_BINS"]]` is INVALID
#     1a. but not all element types are valid: `arr: NDArray[ND.Shape["N_POS_BINS, 4"], float]` is INVALID (`InvalidArgumentsError: Unexpected argument '<class 'float'>', expecting Structure[<StructureExpression>] or Literal[<StructureExpression>] or a dtype or typing.Any.`); while `arr: NDArray[ND.Shape["N_POS_BINS, 4"], np.floating]` is valid, `arr: NDArray[ND.Shape["N_POS_BINS, 4"], np.float_]` is valid
# 
# 2. You must start each labeled dimension variable with an *upper-case letter* -- `arr1: NDArray[ND.Shape["Size1, Size2"], Any]` is valid, `arr1: NDArray[ND.Shape["size1, size2"], Any]` is INVALID
# 3. The contents of `ND.Shape["Size1, Size2"]` must always be a single string. It would be far more natural for me to write `ND.Shape["Size1", "Size2"]` (a list of strings) -- `ND.Shape["Size1, Size2"]` is valid while `ND.Shape["Size1", "Size2"]` is INVALID
# 
# ## Examples of valid shapes:
# ```python
# arr: NDArray[ND.Shape["1, N_TIME_BINS, 4"], np.uint8]
# arr: NDArray[ND.Shape["Size, Size"], Any] = np.random.randn(2, 2)
# arr: NDArray[ND.Shape["Size1, Size2"], Any] = np.random.randn(2, 2)
# arr: NDArray[ND.Shape["Size1, Size2"], Any] = np.random.randn(2, 2) # Case-sensitivity: valid
# # arr: NDArray[ND.Shape["Size1, size2"], Any] =  np.random.randn(2, 2) # Case-sensitivity: InvalidShapeError: 'Size1, size2' is not a valid shape expression.
# ```
# 
# 
# 
# 

# %%
import nptyping as ND
from nptyping import NDArray



# %%


# %% [markdown]
# # NDArray formatting

# %%
# Pho's Formatting Preferences
import builtins

import IPython
from IPython.core.formatters import PlainTextFormatter
from IPython import get_ipython

from pyphocorehelpers.preferences_helpers import set_pho_preferences, set_pho_preferences_concise, set_pho_preferences_verbose
set_pho_preferences_concise()
# Jupyter-lab enable printing for any line on its own (instead of just the last one in the cell)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# BEGIN PPRINT CUSTOMIZATION ___________________________________________________________________________________________ #

## IPython pprint
from pyphocorehelpers.pprint import wide_pprint, wide_pprint_ipython, wide_pprint_jupyter, MAX_LINE_LENGTH
# Override default pprint
builtins.pprint = wide_pprint

ip = get_ipython()

from pyphocorehelpers.ipython_helpers import CustomFormatterMagics

# Register the magic
get_ipython().register_magics(CustomFormatterMagics)

# from pho_jupyter_preview_widget.display_helpers import array_repr_with_graphical_preview
# from pho_jupyter_preview_widget.ipython_helpers import PreviewWidgetMagics

# # Register the magic
# ip.register_magics(PreviewWidgetMagics)

# # %config_ndarray_preview width=500

# # Register the custom display function for NumPy arrays
# # ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_graphical_shape_repr_html(arr))
# # ip = array_repr_with_graphical_shape(ip=ip)
# ip = array_repr_with_graphical_preview(ip=ip)
# # ip = dataframe_show_more_button(ip=ip)

text_formatter: PlainTextFormatter = ip.display_formatter.formatters['text/plain']
text_formatter.max_width = MAX_LINE_LENGTH
text_formatter.for_type(object, wide_pprint_jupyter)


# END PPRINT CUSTOMIZATION ___________________________________________________________________________________________ #

# %% [markdown]
# # Beautiful Flexible 'pd.DataFrame' column access considering synonyms

# %%
time_col: str = TimeColumnAliasesProtocol.find_first_extant_suitable_columns_name(df, col_connonical_name='start', required_columns_synonym_dict={"start":{'begin','start_t','ripple_start_t'}, "stop":['end','stop_t']}, should_raise_exception_on_fail=False)
assert time_col in df


# %% [markdown]
# # <a id='toc20_'></a>[#TODO 2025-03-04 17:17: - [ ] Documentation for all `DirectionalPlacefieldGlobalComputationFunction.py` classes](#toc0_)
# 
# # General Classes ____________________________________________________________________________________________________ #
# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import BasePositionDecoder
# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import DecodedFilterEpochsResult
# from pyphoplacecellanalysis.Analysis.Decoder.reconstruction import SingleEpochDecodedResult
# 
# 
# # Specialty Classes __________________________________________________________________________________________________ #
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import TrackTemplates
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalLapsResult
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPseudo2DDecodersResult
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalDecodersContinuouslyDecodedResult
# 

# %% [markdown]
# 

# %% [markdown]
# # 2025-04-14 - Conform

# %% [markdown]
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Display.py:681](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Display.py:681)
# ```python
# # From `General.Pipeline.Stages.Display.conform`
# @classmethod
# def conform(cls, obj):
#     """ makes the object conform to this mixin by adding its properties. 
#     Usage:
#         from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import PipelineWithComputedPipelineStageMixin, ComputedPipelineStage
#         from pyphoplacecellanalysis.General.Pipeline.Stages.Display import PipelineWithDisplayPipelineStageMixin, PipelineWithDisplaySavingMixin
#         from pyphoplacecellanalysis.General.Pipeline.Stages.Filtering import FilteredPipelineMixin
#         from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import PipelineWithInputStage, PipelineWithLoadableStage
#         from pyphoplacecellanalysis.General.Pipeline.Stages.BaseNeuropyPipelineStage import PipelineStage
#         from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline
# 
#         PipelineWithDisplaySavingMixin.conform(curr_active_pipeline)
# 
#     """
#     def conform_to_implementing_method(func):
#         """ captures 'obj', 'cls'"""
#         setattr(type(obj), func.__name__, func)
#     
#     conform_to_implementing_method(cls.build_display_context_for_session)
#     conform_to_implementing_method(cls.build_display_context_for_filtered_session)
#     # conform_to_implementing_method(cls.write_figure_to_daily_programmatic_session_output_path)
#     # conform_to_implementing_method(cls.write_figure_to_output_path)
#     conform_to_implementing_method(cls.output_figure)
#     
# ```

# %% [markdown]
# # ⚓💯📚 Batch Exports/Output Locations and Generating Functions
# 
# `K:\scratch\collected_outputs`
# 

# %% [markdown]
# ## ⚓💾 Batch Data Exports (CSVs, .pkl, .h5)

# %% [markdown]
# ⚓💾
# 
# ## Computation BATCH_DAY_DATE subfolders:
# `K:\scratch\collected_outputs\2025-07-29_GL`
# 
# ### `*_neuron_replay_stats_df.csv` one for each session
# 
# Exported by `compute_and_export_session_extended_placefield_peak_information_completion_function(...)`:
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py:3094](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/UserCompletionHelpers/batch_user_completion_helpers.py:3094)
# ```python
# # From `General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers.all_neuron_stats_table`
# all_neuron_stats_table: pd.DataFrame = AcrossSessionsResults.build_neuron_identities_df_for_CSV(curr_active_pipeline=curr_active_pipeline)
# ```
# 
# ----
# 
# ## Instantaneous Firing Rates
# 
#     `across_session_result_long_short_recomputed_inst_firing_rate_2025-07-29_1000.0.pkl` one for each firing rate
# 

# %% [markdown]
# 
# ## (FAT) format
# `2025-07-28_0130PM-kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0-(FAT)_tbin-0.025.csv`
# 
# ## `*_neuron_replay_stats_df.csv` exports
# `2025-07-17_GL-2006-4-09_16-40-54_neuron_replay_stats_df.csv`
# 
# ## Instantaneous Firing Rates
# `across_session_result_long_short_recomputed_inst_firing_rate_2025-07-29_1000.0.pkl`

# %% [markdown]
# ### Others (.pkl, .h5)
# 
# #### `(first_spike_activity_data)`
# `2025-06-11-kdiba_pin01_one_fet11-01_12-58-54__withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_5.0-(first_spike_activity_data).h5`
# 
# #### Pipeline Results
# `2025-06-17_GL_2006-6-09_1-22-43_pipeline_results.h5`

# %% [markdown]
# ## Batch Figure Output Location
# Results are saved out to 'gen_scripts/{session_folder}/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/{batch_export_day_date}/{session_context_path_parts}', e.g. 'gen_scripts/run_kdiba_gor01_one_2006-6-08_14-26-15/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2025-07-02/kdiba/gor01/one/2006-6-08_14-26-15'
# 
# **To easily collect figures in a common folder**, do `gen_scripts` batch figure export copy to `collected_figures` folder via `copy_batch_output_figures_to_common_figures_dir(...)`
# 
# ```python
# import shutil
# from pathlib import Path
# from typing import List, Optional
# from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import copy_batch_output_figures_to_common_figures_dir
# 
# ## Move figures
# generate_figures_script_paths = [Path(v) for v in ['C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_one_2006-6-07_11-26-53/figures_kdiba_gor01_one_2006-6-07_11-26-53.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_one_2006-6-08_14-26-15/figures_kdiba_gor01_one_2006-6-08_14-26-15.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_one_2006-6-09_1-22-43/figures_kdiba_gor01_one_2006-6-09_1-22-43.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_one_2006-6-12_15-55-31/figures_kdiba_gor01_one_2006-6-12_15-55-31.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_two_2006-6-07_16-40-19/figures_kdiba_gor01_two_2006-6-07_16-40-19.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_two_2006-6-08_21-16-25/figures_kdiba_gor01_two_2006-6-08_21-16-25.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_two_2006-6-09_22-24-40/figures_kdiba_gor01_two_2006-6-09_22-24-40.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_gor01_two_2006-6-12_16-53-46/figures_kdiba_gor01_two_2006-6-12_16-53-46.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_vvp01_one_2006-4-09_17-29-30/figures_kdiba_vvp01_one_2006-4-09_17-29-30.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_vvp01_one_2006-4-10_12-25-50/figures_kdiba_vvp01_one_2006-4-10_12-25-50.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_vvp01_two_2006-4-09_16-40-54/figures_kdiba_vvp01_two_2006-4-09_16-40-54.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_vvp01_two_2006-4-10_12-58-3/figures_kdiba_vvp01_two_2006-4-10_12-58-3.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_pin01_one_11-02_17-46-44/figures_kdiba_pin01_one_11-02_17-46-44.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_pin01_one_11-02_19-28-0/figures_kdiba_pin01_one_11-02_19-28-0.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_pin01_one_11-03_12-3-25/figures_kdiba_pin01_one_11-03_12-3-25.py',
#  'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/gen_scripts/run_kdiba_pin01_one_fet11-01_12-58-54/figures_kdiba_pin01_one_fet11-01_12-58-54.py']]
# 
# _copied_outputs = copy_batch_output_figures_to_common_figures_dir(generate_figures_script_paths=generate_figures_script_paths)
# _copied_outputs
# ```

# %% [markdown]
# ### Internal Call Hierarchy Documentation

# %% [markdown]
# ##### Nearly all the figures are made via `batch_extended_programmatic_figures(...)`
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py:570](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py:570)
# ```python
# # From `General.Batch.NonInteractiveProcessing.batch_extended_programmatic_figures.curr_active_pipeline`
# def batch_extended_programmatic_figures(curr_active_pipeline, write_vector_format=False, write_png=True, debug_print=False):
# 	""" Generation and display of figures should produce as many as possible, not stopping after failing on one. """
# ```
# 
# ```python
# active_out_figure_paths = curr_active_pipeline.output_figure(extracted_context, fig, write_vector_format=write_vector_format, write_png=write_png, debug_print=debug_print)
# ```

# %% [markdown]
# This internally calls `programmatic_render_to_file(...)` and `programmatic_render_to_pdf(...)`
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:1027](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:1027)
# ```python
# # From `General.Mixins.ExportHelpers.programmatic_render_to_file.curr_active_pipeline`
# def programmatic_render_to_file(curr_active_pipeline, curr_display_function_name='_display_plot_decoded_epoch_slices', subset_includelist=None, subset_excludelist=None, write_vector_format=False, write_png=True, debug_print=False, **kwargs):
#     """ Loops through the individual epochs in a session (e.g. ['maze1', 'maze2', 'maze']) analagous to the structure of `programmatic_display_to_PDF` and programmatically calls `perform_write_to_file` with the appropriate parameters.
#     Newer Programmatic .png and .pdf outputs
#     curr_display_function_name = '_display_plot_decoded_epoch_slices' 
# 
#     Looks it this is 
# ```

# %% [markdown]
# Inside these, the output paths are determined by:
# ```python`
# fig_man = curr_active_pipeline.get_output_manager()
# fig_man
# 
# FileOutputManager(figure_output_location=<FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER: 'daily_programmatic_output_folder'>, context_to_path_mode=<ContextToPathMode.HIERARCHY_UNIQUE: 'hierarchy_unique'>, override_output_parent_path=None)
# ````

# %% [markdown]
# ## Masking "bad" time bins with low firing

# %% [markdown]
# [/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py:1526](vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/reconstruction.py:1526)
# ```python
# # From `Analysis.Decoder.reconstruction.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin`
#         a_decoded_result.p_x_given_n_list[0].shape # (59, 2, 69487)
#         a_decoded_result.most_likely_position_indicies_list[0].shape # .shape (2, 69487)
#         a_decoded_result.most_likely_positions_list[0].shape # .shape (69487, 2)
# 
# 
#         spikes_df: pd.DataFrame = deepcopy(get_proper_global_spikes_df(curr_active_pipeline))
#         non_PBE_all_directional_pf1D_Decoder, pseudo2D_continuous_specific_decoded_result, continuous_decoded_results_dict, non_PBE_marginal_over_track_ID, (time_bin_containers, time_window_centers) = nonPBE_results._build_merged_joint_placefields_and_decode(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
#         maksed_pseudo2D_continuous_specific_decoded_result, mask_index_tuple = pseudo2D_continuous_specific_decoded_result.mask_computed_DecodedFilterEpochsResult_by_required_spike_counts_per_time_bin(spikes_df=deepcopy(get_proper_global_spikes_df(curr_active_pipeline)))
#         # (is_time_bin_active_list, inactive_mask_list, all_time_bin_indicies_list, last_valid_indices_list) = mask_index_tuple
#         maksed_pseudo2D_continuous_specific_decoded_result
# ```

# %%
should_add_col_row_labels

# %% [markdown]
# # Posterior Image Exports

# %% [markdown]
# ## 2025-08-13 - RELEVANT FOR Figure 4 Publication Examples
# 
# "K:/scratch/collected_outputs/figures/_temp_individual_posteriors/2025-08-13/gor01_one_2006-6-09_1-22-43/ripple/combined/multi"
# [_temp_individual_posteriors ... combined/multi Folder](K:/scratch/collected_outputs/figures/_temp_individual_posteriors/2025-08-13/gor01_one_2006-6-09_1-22-43/ripple/combined/multi)
# 
# #### Produced by 
# ```python
# PosteriorExporting.post_export_build_combined_images(out_custom_formats_dict=out_custom_formats_dict)
# ```
# which can be made in batch by calling `'figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function'` with `included_figures_names=['_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay']` specified
# 
# 
# ![alt text](p_x_given_n[004].png)

# %% [markdown]
# ### Other Posterior Image Exports
# <!-- [_temp_individual_posteriors Folder](K:/scratch/collected_outputs/figures/_temp_individual_posteriors) -->
# 
# C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\output\array_to_images
# 

# %% [markdown]
# [_temp_individual_posteriors](command:revealFileInOS?%22K:/scratch/collected_outputs/figures/_temp_individual_posteriors%22)

# %%
## OLD? `_display_directional_merged_pf_decoded_stacked_epoch_slices`

# %%
from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.UserCompletionHelpers.batch_user_completion_helpers import figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function, SimpleBatchComputationDummy
from pyphoplacecellanalysis.Analysis.Decoder.context_dependent import GenericDecoderDictDecodedEpochsDictResult #, KnownNamedDecoderTrainedComputeEpochsType, KnownNamedDecodingEpochsType, MaskedTimeBinFillType, DataTimeGrain, GenericResultTupleIndexType
a_dummy = SimpleBatchComputationDummy(BATCH_DATE_TO_USE, collected_outputs_path, True)

curr_active_pipeline.reload_default_display_functions()

## Settings:
_across_session_results_extended_dict = {}

complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()

_across_session_results_extended_dict = _across_session_results_extended_dict | figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(a_dummy, None,
                                                    curr_session_context=complete_session_context,
                                                    curr_session_basedir=curr_active_pipeline.sess.basepath.resolve(), curr_active_pipeline=curr_active_pipeline,
                                                    included_figures_names=['_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay'], ## This seems sufficient
                                                    fail_on_exception_for_debugging=True,
                                                )

# _across_session_results_extended_dict

# %% [markdown]
# # Python Value Display/Formatting

# %% [markdown]
# How can I modify my custom object's __repr__ so that it prints in a more readible format, e.g.:
# ## Current Output:
# Renders a single line
# ```python
# PhoUIContainer({'name': 'PhoUIContainer', 'layout': <PyQt5.QtWidgets.QGridLayout object at 0x00000202084A39D0>, 'main_time_curves_view_widget': <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.PlotItem.PlotItem.PlotItem object at 0x0000020208646550>, 'main_time_curves_view_legend': <pyphoplacecellanalysis.External.pyqtgraph.graphicsItems.LegendItem.LegendItem object at 0x0000020284693160>, 'connections': ConnectionsContainer({'tracks': {}, 'intervals': <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAF510>, 'rasters[raster_window]': <PyQt5.QtCore.QMetaObject.Connection object at 0x0000020398415E40>}), 'wrapper_widget': <PyQt5.QtWidgets.QWidget object at 0x00000202084A3820>, 'wrapper_layout': <PyQt5.QtWidgets.QVBoxLayout object at 0x00000202084A3790>, 'dynamic_docked_widget_container': <pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget.NestedDockAreaWidget object at 0x00000202084A3A60>, 'matplotlib_view_widgets': {'interval_overview': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002020CA8F280>, 'intervals': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002020CAAAB80>, 'rasters[raster_overview]': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002020CAB4C10>, 'rasters[raster_window]': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002027D09C430>, 'new_curves_separate_plot': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x0000020208646F70>}, 'scroll_window_region': <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem.CustomLinearRegionItem object at 0x000002027D09CE50>, 'menus': PhoUIContainer({'name': 'PhoUIContainer', 'custom_context_menus': PhoUIContainer({'name': 'PhoUIContainer', 'add_renderables': (<pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable.LocalMenus_AddRenderable object at 0x00000202083FF700>, <PyQt5.QtWidgets.QMenu object at 0x0000020208416700>, ([<PyQt5.QtWidgets.QAction object at 0x00000202086EA0D0>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA1F0>, <PyQt5.QtWidgets.QAction object at 0x00000202088929D0>, <PyQt5.QtWidgets.QAction object at 0x0000020208892EE0>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA160>, <PyQt5.QtWidgets.QAction object at 0x0000020208771CA0>, <PyQt5.QtWidgets.QAction object at 0x0000020208771C10>, <PyQt5.QtWidgets.QAction object at 0x0000020208892CA0>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA040>], [<function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202087715E0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208771700>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202087718B0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202087719D0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208771E50>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EACA0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EA670>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EA310>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EA430>], [<PyQt5.QtCore.QMetaObject.Connection object at 0x0000020208490F20>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000202978AB5F0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000202978AB2E0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAF580>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFE40>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAF660>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFEB0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFD60>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFF90>]), ([<PyQt5.QtWidgets.QAction object at 0x00000202086EA790>, <PyQt5.QtWidgets.QAction object at 0x0000020208892D30>, <PyQt5.QtWidgets.QAction object at 0x0000020208892F70>, <PyQt5.QtWidgets.QAction object at 0x0000020208771F70>, <PyQt5.QtWidgets.QAction object at 0x0000020208892C10>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA820>], [<function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202084165E0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892A60>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202088925E0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892550>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202088924C0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892430>], [<PyQt5.QtCore.QMetaObject.Connection object at 0x0000020398415EB0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAC9900>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB040>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB0B0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB120>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB190>]), ([<PyQt5.QtWidgets.QAction object at 0x0000020208771670>, <PyQt5.QtWidgets.QAction object at 0x0000020208771430>], [<function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202088923A0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892310>], [<PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB200>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB270>]))})}), 'epochs_render_configs_widget': <pyphoplacecellanalysis.GUI.Qt.Widgets.EpochRenderConfigWidget.EpochRenderConfigWidget.EpochRenderConfigsListWidget object at 0x000002027F428550>})
# ```
# 
# ## Desired Output:
# Renders as a visually nested tree
# ```python
# PhoUIContainer({'name': 'PhoUIContainer', 
# 				'layout': QGridLayout, 
# 				'main_time_curves_view_widget': pg.PlotItem, 
# 				'main_time_curves_view_legend': pg.LegendItem, 
# 				'connections': ConnectionsContainer({'tracks': {}, 'intervals': <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAF510>, 'rasters[raster_window]': <PyQt5.QtCore.QMetaObject.Connection object at 0x0000020398415E40>}), 'wrapper_widget': <PyQt5.QtWidgets.QWidget object at 0x00000202084A3820>, 'wrapper_layout': <PyQt5.QtWidgets.QVBoxLayout object at 0x00000202084A3790>, 'dynamic_docked_widget_container': <pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget.NestedDockAreaWidget object at 0x00000202084A3A60>, 'matplotlib_view_widgets': {'interval_overview': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002020CA8F280>, 'intervals': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002020CAAAB80>, 'rasters[raster_overview]': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002020CAB4C10>, 'rasters[raster_window]': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x000002027D09C430>, 'new_curves_separate_plot': <pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.PyqtgraphTimeSynchronizedWidget.PyqtgraphTimeSynchronizedWidget object at 0x0000020208646F70>}, 'scroll_window_region': <pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.CustomLinearRegionItem.CustomLinearRegionItem object at 0x000002027D09CE50>,
# 				 'menus': PhoUIContainer({'name': 'PhoUIContainer', 'custom_context_menus': PhoUIContainer({'name': 'PhoUIContainer', 'add_renderables': (<pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable.LocalMenus_AddRenderable.LocalMenus_AddRenderable object at 0x00000202083FF700>, <PyQt5.QtWidgets.QMenu object at 0x0000020208416700>, ([<PyQt5.QtWidgets.QAction object at 0x00000202086EA0D0>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA1F0>, <PyQt5.QtWidgets.QAction object at 0x00000202088929D0>, <PyQt5.QtWidgets.QAction object at 0x0000020208892EE0>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA160>, <PyQt5.QtWidgets.QAction object at 0x0000020208771CA0>, <PyQt5.QtWidgets.QAction object at 0x0000020208771C10>, <PyQt5.QtWidgets.QAction object at 0x0000020208892CA0>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA040>], [<function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202087715E0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208771700>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202087718B0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202087719D0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208771E50>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EACA0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EA670>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EA310>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202086EA430>], [<PyQt5.QtCore.QMetaObject.Connection object at 0x0000020208490F20>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000202978AB5F0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000202978AB2E0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAF580>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFE40>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAF660>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFEB0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFD60>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAAFF90>]), ([<PyQt5.QtWidgets.QAction object at 0x00000202086EA790>, <PyQt5.QtWidgets.QAction object at 0x0000020208892D30>, <PyQt5.QtWidgets.QAction object at 0x0000020208892F70>, <PyQt5.QtWidgets.QAction object at 0x0000020208771F70>, <PyQt5.QtWidgets.QAction object at 0x0000020208892C10>, <PyQt5.QtWidgets.QAction object at 0x00000202086EA820>], [<function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202084165E0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892A60>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202088925E0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892550>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202088924C0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892430>], [<PyQt5.QtCore.QMetaObject.Connection object at 0x0000020398415EB0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x000002020CAC9900>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB040>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB0B0>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB120>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB190>]), ([<PyQt5.QtWidgets.QAction object at 0x0000020208771670>, <PyQt5.QtWidgets.QAction object at 0x0000020208771430>], [<function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x00000202088923A0>, <function LocalMenus_AddRenderable._build_renderable_menu.<locals>.<lambda> at 0x0000020208892310>], [<PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB200>, <PyQt5.QtCore.QMetaObject.Connection object at 0x00000203BBFAB270>]))})}), 'epochs_render_configs_widget': <pyphoplacecellanalysis.GUI.Qt.Widgets.EpochRenderConfigWidget.EpochRenderConfigWidget.EpochRenderConfigsListWidget object at 0x000002027F428550>})
# ```
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# 

# %% [markdown]
# # Snapshotted Timeline for displaying 2D Decoded Positions


