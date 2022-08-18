## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from matplotlib.colors import ListedColormap
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt

from pyphoplacecellanalysis.General.Configs.DynamicConfigs import PlottingConfig, InteractivePlaceCellConfig
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange

should_force_recompute_placefields = True
should_display_2D_plots = True

# ==================================================================================================================== #
# 2022-08-18                                                                                                           #
# ==================================================================================================================== #

from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
from pyphoplacecellanalysis.GUI.Qt.DecoderPlotSelectorControls.DecoderPlotSelectorWidget import DecoderPlotSelectorWidget # for context_nested_docks
from pyphoplacecellanalysis.GUI.Qt.FigureFormatConfigControls.FigureFormatConfigControls import FigureFormatConfigControls # for context_nested_docks
_debug_print = True

def single_context_nested_docks(curr_active_pipeline, active_config_name, app, master_dock_win, debug_print=True):
        out_display_items = dict()
        
        # Get relevant variables for this particular context:
        # curr_active_pipeline is set above, and usable here
        sess = curr_active_pipeline.filtered_sessions[active_config_name]

        # active_computation_results = curr_active_pipeline.computation_results[active_config_name]
        # active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
        # active_computation_config = curr_active_pipeline.computation_results[active_config_name].computation_config
        # active_computation_errors = curr_active_pipeline.computation_results[active_config_name].accumulated_errors
        # active_pf_1D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
        # active_pf_2D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D']    
        # active_pf_1D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_dt', None)
        # active_pf_2D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_dt', None)
        # active_firing_rate_trends = curr_active_pipeline.computation_results[active_config_name].computed_data.get('firing_rate_trends', None)
        active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None)
        # active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
        # active_extended_stats = curr_active_pipeline.computation_results[active_config_name].computed_data.get('extended_stats', None)
        # active_eloy_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('EloyAnalysis', None)
        # active_simpler_pf_densities_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('SimplerNeuronMeetingThresholdFiringAnalysis', None)
        # active_ratemap_peaks_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', None)
        # active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
        # active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()
        # curr_spikes_df = sess.spikes_df

        curr_active_config = curr_active_pipeline.active_configs[active_config_name]
        # curr_active_display_config = curr_active_config.plotting_config

        ## Build the active context by starting with the session context:
        active_identifying_session_ctx = sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'
        ## Add the filter to the active context
        active_identifying_session_ctx.add_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

        def on_finalize_figure_format_config(updated_figure_format_config):
                if debug_print:
                    print('on_finalize_figure_format_config')
                    print(f'\t {updated_figure_format_config}')
                # figure_format_config = updated_figure_format_config
                pass
                
        ## Finally, add the display function to the active context
        active_identifying_ctx = active_identifying_session_ctx.adding_context('display_fn', display_fn_name='figure_format_config_widget')
        active_identifying_ctx_string = active_identifying_ctx.get_description(separator='|') # Get final discription string:
        if debug_print:
            print(f'active_identifying_ctx_string: {active_identifying_ctx_string}')

        figure_format_config_widget = FigureFormatConfigControls(config=curr_active_config)
        figure_format_config_widget.figure_format_config_finalized.connect(on_finalize_figure_format_config)
        figure_format_config_widget.show() # even without .show() being called, the figure still appears
        ## Get the figure_format_config from the figure_format_config widget:
        figure_format_config = figure_format_config_widget.figure_format_config

        master_dock_win.add_display_dock(identifier=active_identifying_ctx_string, widget=figure_format_config_widget, dockIsClosable=False)
        out_display_items[active_identifying_ctx] = (figure_format_config_widget)

        
        ## Finally, add the display function to the active context
        active_identifying_ctx = active_identifying_session_ctx.adding_context('display_fn', display_fn_name='2D Position Decoder')
        active_identifying_ctx_string = active_identifying_ctx.get_description(separator='|') # Get final discription string:
        if debug_print:
            print(f'active_identifying_ctx_string: {active_identifying_ctx_string}')
        decoder_plot_widget = DecoderPlotSelectorWidget()
        decoder_plot_widget.show()
        master_dock_win.add_display_dock(identifier=active_identifying_ctx_string, widget=decoder_plot_widget, dockIsClosable=False)
        out_display_items[active_identifying_ctx] = (decoder_plot_widget)

        # Get the decoders from the computation result:
        # active_one_step_decoder = computation_result.computed_data['pf2D_Decoder'] # doesn't actually require the Decoder, could just use computation_result.computed_data['pf2D']            
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        active_identifying_ctx = active_identifying_session_ctx.adding_context('display_fn', display_fn_name='_temp_pyqtplot_plot_image_array')
        active_identifying_ctx_string = active_identifying_ctx.get_description(separator='|') # Get final discription string:
        if debug_print:
            print(f'active_identifying_ctx_string: {active_identifying_ctx_string}')
            
        ## Build the widget:
        app, pyqtplot_pf2D_parent_root_widget, pyqtplot_pf2D_root_render_widget, pyqtplot_pf2D_plot_array, pyqtplot_pf2D_img_item_array, pyqtplot_pf2D_other_components_array = _temp_pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy, app=app, parent_root_widget=None, root_render_widget=None, max_num_columns=8)
        pyqtplot_pf2D_parent_root_widget.show()
        master_dock_win.add_display_dock(identifier=active_identifying_ctx_string, widget=pyqtplot_pf2D_parent_root_widget, dockIsClosable=False)
        out_display_items[active_identifying_ctx] = (pyqtplot_pf2D_parent_root_widget, pyqtplot_pf2D_root_render_widget, pyqtplot_pf2D_plot_array, pyqtplot_pf2D_img_item_array, pyqtplot_pf2D_other_components_array)
        
        return active_identifying_session_ctx, out_display_items
        # END single_context_nested_docks(...)
        
        
def context_nested_docks(curr_active_pipeline, debug_print=True):
    active_config_names = curr_active_pipeline.active_completed_computation_result_names # ['maze', 'sprinkle']
    
    master_dock_win, app = DockAreaWrapper._build_default_dockAreaWindow(title='active_global_window', defer_show=False)
    master_dock_win.resize(1920, 1200)

    out_items = {}
    for a_config_name in active_config_names:
        active_identifying_session_ctx, out_display_items = single_context_nested_docks(curr_active_pipeline=curr_active_pipeline, active_config_name=a_config_name, app=app, master_dock_win=master_dock_win, debug_print=debug_print)
        out_items[a_config_name] = (active_identifying_session_ctx, out_display_items)
        
    return master_dock_win, app, out_items

import matplotlib.pyplot as plt 
from matplotlib.widgets import Slider # needed for _temp_debug_two_step_plots_animated_imshow

## Copied from DecoderPredictionError.py to modify with adding nearest animal position at each timestep:
def _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, time_binned_position_df, variable_name='p_x_given_n_and_x_prev', override_variable_value=None, update_callback_function=None):
    """Matplotlib-based imshow plot with interactive slider for displaying two-step bayesian decoding results

    ## Added _update_measured_animal_position_point(...)
    DEPENDS ON active_computed_data.extended_stats.time_binned_position_df
    
    Args:
        active_one_step_decoder ([type]): [description]
        active_two_step_decoder ([type]): [description]
        time_binned_position_df: should be obtained from `active_computed_data.extended_stats.time_binned_position_df` by default
        variable_name (str, optional): [description]. Defaults to 'p_x_given_n_and_x_prev'.
        override_variable_value ([type], optional): [description]. Defaults to None.
        update_callback_function ([type], optional): [description]. Defaults to None.
        
        
    Usage:
        # Simple plot type 1:
        # plotted_variable_name = kwargs.get('variable_name', 'p_x_given_n') # Tries to get the user-provided variable name, otherwise defaults to 'p_x_given_n'
        plotted_variable_name = 'p_x_given_n' # Tries to get the user-provided variable name, otherwise defaults to 'p_x_given_n'
        _temp_debug_two_step_plots_animated_imshow(active_one_step_decoder, active_two_step_decoder, active_computed_data.extended_stats.time_binned_position_df, variable_name=plotted_variable_name) # Works

    """
    if override_variable_value is None:
        try:
            variable_value = active_two_step_decoder[variable_name]
        except (TypeError, KeyError):
            # fallback to the one_step_decoder
            variable_value = getattr(active_one_step_decoder, variable_name, None)
    else:
        # if override_variable_value is set, ignore the input info and use it.
        variable_value = override_variable_value

    num_frames = np.shape(variable_value)[-1]
    debug_print = False
    if debug_print:
        print(f'_temp_debug_two_step_plots_animated_imshow: variable_name="{variable_name}", np.shape: {np.shape(variable_value)}, num_frames: {num_frames}')

    fig, ax = plt.subplots(ncols=1, nrows=1, num=f'debug_two_step_animated: variable_name={variable_name}', figsize=(15,15), clear=True, constrained_layout=False)
    plt.subplots_adjust(left=0.25, bottom=0.25)

    frame = 0
    
    # Get extents:    
    xmin, xmax, ymin, ymax = (active_one_step_decoder.xbin[0], active_one_step_decoder.xbin[-1], active_one_step_decoder.ybin[0], active_one_step_decoder.ybin[-1])
    x_first_extent = (xmin, xmax, ymin, ymax) # traditional order of the extant axes
    active_extent = x_first_extent # for 'x == horizontal orientation'
    # active_extent = y_first_extent # for 'x == vertical orientation'

    main_plot_kwargs = {
        'origin': 'lower',
        'cmap': 'turbo',
        'extent': active_extent,
        # 'aspect':'auto',
    }

    curr_val = variable_value[:,:,frame] # untranslated output:
    curr_val = np.swapaxes(curr_val, 0, 1) # x_horizontal_matrix: swap the first two axes while leaving the last intact. Returns a view into the matrix so it doesn't modify the value
    
    im_out = ax.imshow(curr_val, **main_plot_kwargs)
    
    ## Setup Auxillary Plots:
    active_resampled_pos_df = time_binned_position_df.copy() # active_computed_data.extended_stats.time_binned_position_df  # 1717 rows × 16 columns
    active_resampled_measured_positions = active_resampled_pos_df[['x','y']].to_numpy() # The measured positions resampled (interpolated) at the window centers. 
    measured_point = np.squeeze(active_resampled_measured_positions[frame,:])
    ## decided on using scatter
    # measured_positions_scatter = ax.scatter(measured_point[0], measured_point[1], color='white') # PathCollection
    measured_positions_scatter, = ax.plot(measured_point[0], measured_point[1], color='white', marker='o', ls='') # PathCollection
    
    def _update_measured_animal_position_point(time_window_idx, ax=None):
        """ captures `active_resampled_measured_positions` and `measured_positions_scatter` """
        measured_point = np.squeeze(active_resampled_measured_positions[time_window_idx,:])
        ## TODO: this would need to use set_offsets(...) if we wanted to stick with scatter plot.
        measured_positions_scatter.set_xdata(measured_point[0])
        measured_positions_scatter.set_ydata(measured_point[1])
    
    # for 'x == horizontal orientation':
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # ax.axis("off")
    plt.title(f'debug_two_step: {variable_name}')

    axcolor = 'lightgoldenrodyellow'
    axframe = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    sframe = Slider(axframe, 'Frame', 0, num_frames-1, valinit=2, valfmt='%d') # MATPLOTLIB Slider

    def update(val):
        new_frame = int(np.around(sframe.val))
        # print(f'new_frame: {new_frame}')
        curr_val = variable_value[:,:,new_frame] # untranslated output:
        curr_val = np.swapaxes(curr_val, 0, 1) # x_horizontal_matrix: swap the first two axes while leaving the last intact. Returns a view into the matrix so it doesn't modify the value
        im_out.set_data(curr_val)
        # ax.relim()
        # ax.autoscale_view()
        _update_measured_animal_position_point(new_frame, ax=ax)
        
        if update_callback_function is not None:
            update_callback_function(new_frame, ax=ax)
        plt.draw()

    sframe.on_changed(update)
    plt.draw()
    # plt.show()
    
    

from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow # required for display_all_eloy_pf_density_measures_results

def display_all_eloy_pf_density_measures_results(active_pf_2D, active_eloy_analysis, active_simpler_pf_densities_analysis, active_peak_prominence_2d_results):
    """ 
    Usage:
        out_all_eloy_pf_density_fig = display_all_eloy_pf_density_measures_results(active_pf_2D, active_eloy_analysis, active_simpler_pf_densities_analysis, active_peak_prominence_2d_results)
        
    """
    # active_xbins = active_pf_2D.xbin
    # active_ybins = active_pf_2D.ybin
    
    # # *bin_indicies:
    # xbin_indicies = active_pf_2D.xbin_labels -1
    # ybin_indicies = active_pf_2D.ybin_labels -1
    # active_xbins = xbin_indicies
    # active_ybins = ybin_indicies
    
    # *bin_centers: these seem to work
    active_xbins = active_pf_2D.xbin_centers
    active_ybins = active_pf_2D.ybin_centers
    
    out = BasicBinnedImageRenderingWindow(active_eloy_analysis.avg_2D_speed_per_pos, active_xbins, active_ybins, name='avg_velocity', title="Avg Velocity per Pos (X, Y)", variable_label='Avg Velocity')
    out.add_data(row=2, col=0, matrix=active_eloy_analysis.pf_overlapDensity_2D, xbins=active_xbins, ybins=active_ybins, name='pf_overlapDensity', title='pf overlapDensity metric', variable_label='pf overlapDensity')
    out.add_data(row=3, col=0, matrix=active_pf_2D.ratemap.occupancy, xbins=active_xbins, ybins=active_ybins, name='occupancy_seconds', title='Seconds Occupancy', variable_label='seconds')
    out.add_data(row=4, col=0, matrix=active_simpler_pf_densities_analysis.n_neurons_meeting_firing_critiera_by_position_bins_2D, xbins=active_xbins, ybins=active_ybins, name='n_neurons_meeting_firing_critiera_by_position_bins_2D', title='# neurons > 1Hz per Pos (X, Y)', variable_label='# neurons')
    # out.add_data(row=5, col=0, matrix=active_peak_prominence_2d_results.peak_counts.raw, xbins=active_pf_2D.xbin_labels, ybins=active_pf_2D.ybin_labels, name='pf_peak_counts_map', title='# pf peaks per Pos (X, Y)', variable_label='# pf peaks')
    # out.add_data(row=6, col=0, matrix=active_peak_prominence_2d_results.peak_counts.gaussian_blurred, xbins=active_pf_2D.xbin_labels, ybins=active_pf_2D.ybin_labels, name='pf_peak_counts_map_blurred gaussian', title='Gaussian blurred # pf peaks per Pos (X, Y)', variable_label='Gaussian blurred # pf peaks')
    out.add_data(row=5, col=0, matrix=active_peak_prominence_2d_results.peak_counts.raw, xbins=active_xbins, ybins=active_ybins, name='pf_peak_counts_map', title='# pf peaks per Pos (X, Y)', variable_label='# pf peaks')
    out.add_data(row=6, col=0, matrix=active_peak_prominence_2d_results.peak_counts.gaussian_blurred, xbins=active_xbins, ybins=active_ybins, name='pf_peak_counts_map_blurred gaussian', title='Gaussian blurred # pf peaks per Pos (X, Y)', variable_label='Gaussian blurred # pf peaks')

    return out
    

# ==================================================================================================================== #
# Pre 2022-08-17                                                                                                           #
# ==================================================================================================================== #

import pyphoplacecellanalysis.External.pyqtgraph as pg # required for _temp_pyqtplot_plot_image_array
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import _pyqtplot_build_image_bounds_extent # required for _temp_pyqtplot_plot_image_array
from pyphocorehelpers.indexing_helpers import compute_paginated_grid_config # required for _temp_pyqtplot_plot_image_array
from pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_basic import pyqtplot_common_setup # required for _temp_pyqtplot_plot_image_array


def _temp_pyqtplot_plot_image_array(xbin_edges, ybin_edges, images, occupancy, max_num_columns = 5, drop_below_threshold: float=0.0000001, enable_LUT_Histogram=False, app=None, parent_root_widget=None, root_render_widget=None, debug_print=False):
    """ Plots an array of images provided in 'images' argument
    images should be an nd.array with dimensions like: (10, 63, 63), where (N_Images, X_Dim, Y_Dim)
        or (2, 5, 63, 63), where (N_Rows, N_Cols, X_Dim, Y_Dim)
        
    Example:
        # Get flat list of images:
        images = active_one_step_decoder.ratemap.normalized_tuning_curves # (43, 63, 63)
        # images = active_one_step_decoder.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
        occupancy = active_one_step_decoder.ratemap.occupancy

        app, win, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_one_step_decoder.xbin, active_one_step_decoder.ybin, images, occupancy)
        win.show()
        
    # TODO: COMPATIBILITY: replace compute_paginated_grid_config with standardized `_determine_best_placefield_2D_layout` block (see below):
    
    from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons),
        **overriding_dict_with(lhs_dict={'subplots': (40, 3), 'fig_column_width': 8.0, 'fig_row_height': 1.0, 'resolution_multiplier': 1.0, 'max_screen_figure_size': (None, None), 'last_figure_subplots_same_layout': True, 'debug_print': True}, **figure_format_config)) 

    print(f'nfigures: {nfigures}\ndata_aspect_ratio: {data_aspect_ratio}')
    # Loop through each page/figure that's required:
    for page_fig_ind, page_fig_size, page_grid_size in zip(np.arange(nfigures), page_figure_sizes, page_grid_sizes):
        print(f'\tpage_fig_ind: {page_fig_ind}, page_fig_size: {page_fig_size}, page_grid_size: {page_grid_size}')
        # print(f'\tincluded_combined_indicies_pages: {included_combined_indicies_pages}\npage_grid_sizes: {page_grid_sizes}\npage_figure_sizes: {page_figure_sizes}')
        
    """
    
    # pg.setConfigOptions(imageAxisOrder='row-major')
    root_render_widget, parent_root_widget, app = pyqtplot_common_setup(f'_temp_pyqtplot_plot_image_array: {np.shape(images)}', app=app, parent_root_widget=parent_root_widget, root_render_widget=root_render_widget)
    ## TODO: BUG: this makes a new QMainWindow to hold this item, which is inappropriate if it's to be rendered as a child of another control
    
    
    # Creating a GraphicsLayoutWidget as the central widget
    # if root_render_widget is None:
        # root_render_widget = pg.GraphicsLayoutWidget()
    #     parent_root_widget.setCentralWidget(root_render_widget)
        

    pg.setConfigOptions(imageAxisOrder='col-major')
    
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    cmap = pg.colormap.get('jet','matplotlib') # prepare a linear color map

    image_bounds_extent, x_range, y_range = _pyqtplot_build_image_bounds_extent(xbin_edges, ybin_edges, margin=2.0, debug_print=debug_print)
    # image_aspect_ratio, image_width_height_tuple = compute_data_aspect_ratio(x_range, y_range)
    # print(f'image_aspect_ratio: {image_aspect_ratio} - xScale/yScale: {float(image_width_height_tuple.width) / float(image_width_height_tuple.height)}')
    
    # Compute Images:
    included_unit_indicies = np.arange(np.shape(images)[0]) # include all unless otherwise specified
    nMapsToShow = len(included_unit_indicies)

    # Paging Management: Constrain the subplots values to just those that you need
    subplot_no_pagination_configuration, included_combined_indicies_pages, page_grid_sizes = compute_paginated_grid_config(nMapsToShow, max_num_columns=max_num_columns, max_subplots_per_page=None, data_indicies=included_unit_indicies, last_figure_subplots_same_layout=True)   
    page_idx = 0 # page_idx is zero here because we only have one page:
    
    img_item_array = []
    other_components_array = []
    plot_array = []

    for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
        # Need to convert to page specific:
        curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
        curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
        curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
        is_first_column = (curr_page_relative_col == 0)
        is_first_row = (curr_page_relative_row == 0)
        is_last_column = (curr_page_relative_col == (page_grid_sizes[page_idx].num_columns-1))
        is_last_row = (curr_page_relative_row == (page_grid_sizes[page_idx].num_rows-1))
        if debug_print:
            print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')

        neuron_IDX = curr_included_unit_index
        curr_cell_identifier_string = f'Cell[{neuron_IDX}]'
        curr_plot_identifier_string = f'pyqtplot_plot_image_array.{curr_cell_identifier_string}'

        image = np.squeeze(images[a_linear_index,:,:])
        # Pre-filter the data:
        with np.errstate(divide='ignore', invalid='ignore'):
            image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
            if drop_below_threshold is not None:
                image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy

        # Build the image item:
        img_item = pg.ImageItem(image=image, levels=(0,1))
        
        # # plot mode:
        curr_plot = root_render_widget.addPlot(row=curr_row, col=curr_col, title=curr_cell_identifier_string) # old: , name=curr_plot_identifier_string 
        # PERFORMANCE: primary performance bottleneck occurs here, specifically in GraphicsLayout
        ## PERFORMANCE: It's specifically the initialization of PlotItem within addPlot
            # register
                # updateViewLists: called many times, responsible for majority of time in register
                    ## BREAKTHROUGH: register is only called when setting the name kwarg 
                        # if name is not None:
                        #     self.vb.register(name)
            
        curr_plot.showAxes(False)
        if is_last_row:
            curr_plot.showAxes('x', True)
            curr_plot.showAxis('bottom', show=True)
        else:
            curr_plot.showAxes('x', False)
            curr_plot.showAxis('bottom', show=False)
            
        if is_first_column:
            curr_plot.showAxes('y', True)
            curr_plot.showAxis('left', show=True)
        else:
            curr_plot.showAxes('y', False)
            curr_plot.showAxis('left', show=False)
        
        curr_plot.hideButtons() # Hides the auto-scale button
        
        curr_plot.addItem(img_item, defaultPadding=0.0)  # add ImageItem to PlotItem
        # curr_plot.setAspectLocked(lock=True, ratio=image_aspect_ratio)
        # curr_plot.showAxes(True)
        # curr_plot.showGrid(True, True, 0.7)
        # curr_plot.setLabel('bottom', "Label to test offset")
        
        # # Overlay cell identifier text:
        # curr_label = pg.TextItem(f'Cell[{neuron_IDX}]', color=(230, 230, 230))
        # curr_label.setPos(30, 60)
        # curr_label.setParentItem(img_item)
        # # curr_plot.addItem(curr_label, ignoreBounds=True)
        # curr_plot.addItem(curr_label)

        # Update the image:
        img_item.setImage(image, rect=image_bounds_extent, autoLevels=False) # rect: [x, y, w, h]
        img_item.setLookupTable(cmap.getLookupTable(nPts=256), update=False)

        # curr_plot.set
        # margin = 2.0
        # curr_plot.setXRange(global_min_x-margin, global_max_x+margin)
        # curr_plot.setYRange(global_min_y-margin, global_max_y+margin)
        # curr_plot.setXRange(*x_range)
        # curr_plot.setYRange(*y_range)
        curr_plot.setRange(xRange=x_range, yRange=y_range, padding=0.0, update=False, disableAutoRange=True)
        # Sets only the panning limits:
        curr_plot.setLimits(xMin=x_range[0], xMax=x_range[-1], yMin=y_range[0], yMax=y_range[-1])
        # Link Axes to previous item:
        if a_linear_index > 0:
            prev_plot_item = plot_array[a_linear_index-1]
            curr_plot.setXLink(prev_plot_item)
            curr_plot.setYLink(prev_plot_item)
                        
        # Interactive Color Bar:
        bar = pg.ColorBarItem(values= (0, 1), colorMap=cmap, width=5, interactive=False) # prepare interactive color bar
        # Have ColorBarItem control colors of img and appear in 'plot':
        bar.setImageItem(img_item, insert_in=curr_plot)

        img_item_array.append(img_item)
        plot_array.append(curr_plot)
        other_components_array.append({'color_bar':bar})
        
    # Post images loop:
    
    enable_show = False
    
    if parent_root_widget is not None:
        if enable_show:
            parent_root_widget.show()
        
        parent_root_widget.setWindowTitle('pyqtplot image array')

    # pg.exec()
    return app, parent_root_widget, root_render_widget, plot_array, img_item_array, other_components_array


# ==================================================================================================================== #
# 🔜👁️‍🗨️ Merging TimeSynchronized Plotters:                                                                         #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.SpikeRasterWidgets.Spike2DRaster import Spike2DRaster
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedOccupancyPlotter import TimeSynchronizedOccupancyPlotter
from pyphoplacecellanalysis.Pho2D.PyQtPlots.TimeSynchronizedPlotters.TimeSynchronizedPlacefieldsPlotter import TimeSynchronizedPlacefieldsPlotter
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper

def _build_combined_time_synchronized_plotters_window(active_pf_2D_dt, fixed_window_duration = 15.0):
    """ Builds a single window with time_synchronized (time-dependent placefield) plotters controlled by a 2DRasterPlot widget.
    
    Usage:
        active_pf_2D_dt.reset()
        active_pf_2D_dt.update(t=45.0, start_relative_t=True)
        all_plotters, root_dockAreaWindow, app = _build_combined_time_synchronized_plotters_window(active_pf_2D_dt, fixed_window_duration = 15.0)
    """
    def _merge_plotters(spike_raster_plt_2d, curr_sync_occupancy_plotter, curr_placefields_plotter):
        # root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(curr_sync_occupancy_plotter, spike_raster_plt_2d, title='All Time Synchronized Plotters')
        # curr_placefields_plotter, dDisplayItem = root_dockAreaWindow.add_display_dock(identifier='Time Dependent Placefields', widget=curr_placefields_plotter, dockAddLocationOpts=['left'])
        root_dockAreaWindow, app = DockAreaWrapper.wrap_with_dockAreaWindow(curr_sync_occupancy_plotter, curr_placefields_plotter, title='All Time Synchronized Plotters')
        spike_raster_plt_2d, dDisplayItem = root_dockAreaWindow.add_display_dock(identifier='Time Dependent Placefields', widget=spike_raster_plt_2d, dockAddLocationOpts=['bottom'])
        
        ## Register the children items as drivables/drivers:
        root_dockAreaWindow.connection_man.register_drivable(curr_sync_occupancy_plotter)
        root_dockAreaWindow.connection_man.register_drivable(curr_placefields_plotter)
        root_dockAreaWindow.connection_man.register_driver(spike_raster_plt_2d)
        # Wire up signals such that time-synchronized plotters are controlled by the RasterPlot2D:
        occupancy_raster_window_sync_connection = root_dockAreaWindow.connection_man.connect_drivable_to_driver(drivable=curr_sync_occupancy_plotter, driver=spike_raster_plt_2d,
                                                               custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
        placefields_raster_window_sync_connection = root_dockAreaWindow.connection_man.connect_drivable_to_driver(drivable=curr_placefields_plotter, driver=spike_raster_plt_2d,
                                                               custom_connect_function=(lambda driver, drivable: pg.SignalProxy(driver.window_scrolled, delay=0.2, rateLimit=60, slot=drivable.on_window_changed_rate_limited)))
        
        return root_dockAreaWindow, app
    
    # Build the 2D Raster Plotter using a fixed window duration
    current_window_start_time = active_pf_2D_dt.last_t - fixed_window_duration
    spike_raster_plt_2d = Spike2DRaster.init_from_independent_data(active_pf_2D_dt.all_time_filtered_spikes_df, window_duration=fixed_window_duration, window_start_time=current_window_start_time,
                                                                   neuron_colors=None, neuron_sort_order=None, application_name='TimeSynchronizedPlotterControlSpikeRaster2D',
                                                                   enable_independent_playback_controller=False, should_show=False,  parent=None) # setting , parent=spike_raster_plt_3d makes a single window
    spike_raster_plt_2d.setWindowTitle('2D Raster Control Window')
    # Update the 2D Scroll Region to the initial value:
    spike_raster_plt_2d.update_scroll_window_region(current_window_start_time, active_pf_2D_dt.last_t, block_signals=False)
    curr_sync_occupancy_plotter = TimeSynchronizedOccupancyPlotter(active_pf_2D_dt)
    curr_placefields_plotter = TimeSynchronizedPlacefieldsPlotter(active_pf_2D_dt)
    
    root_dockAreaWindow, app = _merge_plotters(spike_raster_plt_2d, curr_sync_occupancy_plotter, curr_placefields_plotter)
    return (spike_raster_plt_2d, curr_sync_occupancy_plotter, curr_placefields_plotter), root_dockAreaWindow, app
    

# ==================================================================================================================== #
# 2022-08-16                                                                                                           #
# ==================================================================================================================== #

from neuropy.utils.dynamic_container import overriding_dict_with # used in display_all_pf_2D_pyqtgraph_binned_image_rendering to only get the valid kwargs to pass from the display config
from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, enumTuningMap2DPlotMode, enumTuningMap2DPlotVariables, _determine_best_placefield_2D_layout, _scale_current_placefield_to_acceptable_range
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, add_bin_ticks, build_binned_imageItem

def display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config, debug_print=True):
    """ 2022-08-16 - A fresh implementation of a pf_2D placefield renderer that uses the BasicBinnedImageRenderingWindow subclass. 
    
    Uses the common `_determine_best_placefield_2D_layout(...)` setup so that its returned subplots layout is the same as the matplotlib version in NeuroPy.neuropy.plotting.ratemaps.plot_ratemap_2D(...) (the main Matplotlib version that works)
    
    Usage:
        out_all_pf_2D_pyqtgraph_binned_image_fig = display_all_pf_2D_pyqtgraph_binned_image_rendering(active_pf_2D, figure_format_config)
        
    """
    drop_below_threshold = figure_format_config.get('drop_below_threshold', None) # try to get the 'drop_below_threshold' argument
    # nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons), subplots=(40, 3), fig_column_width=8.0, fig_row_height=1.0, resolution_multiplier=1.0, max_screen_figure_size=(None, None), last_figure_subplots_same_layout=True, debug_print=True)
    nfigures, num_pages, included_combined_indicies_pages, page_grid_sizes, data_aspect_ratio, page_figure_sizes = _determine_best_placefield_2D_layout(xbin=active_pf_2D.xbin, ybin=active_pf_2D.ybin, included_unit_indicies=np.arange(active_pf_2D.ratemap.n_neurons),
        **overriding_dict_with(lhs_dict={'subplots': (40, 3), 'fig_column_width': 8.0, 'fig_row_height': 1.0, 'resolution_multiplier': 1.0, 'max_screen_figure_size': (None, None), 'last_figure_subplots_same_layout': True, 'debug_print': True}, **figure_format_config))
    active_xbins = active_pf_2D.xbin
    active_ybins = active_pf_2D.ybin    
    out = None
    # New page-based version:
    for page_idx in np.arange(num_pages):
        if debug_print:
            print(f'page_idx: {page_idx}')
        for (a_linear_index, curr_row, curr_col, curr_included_unit_index) in included_combined_indicies_pages[page_idx]:
            # Need to convert to page specific:
            curr_page_relative_linear_index = np.mod(a_linear_index, int(page_grid_sizes[page_idx].num_rows * page_grid_sizes[page_idx].num_columns))
            curr_page_relative_row = np.mod(curr_row, page_grid_sizes[page_idx].num_rows)
            curr_page_relative_col = np.mod(curr_col, page_grid_sizes[page_idx].num_columns)
            # print(f'a_linear_index: {a_linear_index}, curr_page_relative_linear_index: {curr_page_relative_linear_index}, curr_row: {curr_row}, curr_col: {curr_col}, curr_page_relative_row: {curr_page_relative_row}, curr_page_relative_col: {curr_page_relative_col}, curr_included_unit_index: {curr_included_unit_index}')
            neuron_IDX = curr_included_unit_index
            # pfmap = active_pf_2D.ratemap.normalized_tuning_curves[a_linear_index]
            # pfmap = active_pf_2D.ratemap.tuning_curves[a_linear_index].copy()
            pfmap = _scale_current_placefield_to_acceptable_range(np.squeeze(active_pf_2D.ratemap.tuning_curves[a_linear_index,:,:]), occupancy=active_pf_2D.occupancy, drop_below_threshold=drop_below_threshold)            
            
            curr_extended_id_string = active_pf_2D.ratemap.get_extended_neuron_id_string(neuron_i=neuron_IDX)
            # ratemap.neuron_extended_ids[neuron_IDX]
            
            if out is None:
                # first iteration only
                out = BasicBinnedImageRenderingWindow(pfmap, active_xbins, active_ybins, name=f'pf[{curr_extended_id_string}]', title=curr_extended_id_string, variable_label=curr_extended_id_string, wants_crosshairs=False, color_bar_mode='each', )
            else:
                out.add_data(row=curr_page_relative_row, col=curr_page_relative_col, matrix=pfmap, xbins=active_xbins, ybins=active_ybins, name=f'pf[{curr_extended_id_string}]', title=curr_extended_id_string, variable_label=curr_extended_id_string)
        
    return out
    
    
    

# ==================================================================================================================== #
# Pre 2022-08-16 Figure Docking                                                                                        #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper, NestedDockAreaWidget

def _build_docked_pf_2D_figures_widget(active_pf_2D_figures, should_nest_figures_on_filter=True, extant_dockAreaWidget=None, debug_print=False):
    """ Combines the active_pf_2D individual figures into a single widget, with each item being docked and modifiable.
    Requies figures to already be created and passed in the appropriate format.
    
    # TODO: On close should close the figure handles that are currently open. Can use figure_manager to do this.
    
    
    # TODO: Shouldnt' this be a widget instead of a function? Maybe it doesn't matter though.
    
    if should_nest_figures_on_filter is True, the figures are docked in a nested dockarea for each filter (e.g. ['maze1', 'maze2']. Otherwise they are returned flat.
        
    Unique to nested:
        all_nested_dock_area_widgets = {}
        all_nested_dock_area_widget_display_items = {}

    NOTE: This builds a brand-new independent dockAreaWindow, with no option to reuse an extant one.

    Usage:
    
        def _display_specified__display_2d_placefield_result_plot_ratemaps_2D(filter_name):
            active_filter_pf_2D_figures = {}
            active_filter_pf_2D_figures['SPIKES_MAPS'] = curr_active_pipeline.display('_display_2d_placefield_result_plot_ratemaps_2D', filter_name, plot_variable=enumTuningMap2DPlotVariables.SPIKES_MAPS, fignum=plots_fig_nums_dict[filter_name][0], **figure_format_config)[0]
            active_filter_pf_2D_figures['TUNING_MAPS'] = curr_active_pipeline.display('_display_2d_placefield_result_plot_ratemaps_2D', filter_name, plot_variable=enumTuningMap2DPlotVariables.TUNING_MAPS, fignum=plots_fig_nums_dict[filter_name][1], **figure_format_config)[0]
            return active_filter_pf_2D_figures

        active_pf_2D_figures = {}
        ## Plots for each maze programmatically:
        for i, filter_name in enumerate(curr_active_pipeline.active_config_names):
            active_pf_2D_figures[filter_name] = _display_specified__display_2d_placefield_result_plot_ratemaps_2D(filter_name=filter_name)

        active_pf_2D_figures
        # {'maze1': {'SPIKES_MAPS': <Figure size 1728x1080 with 88 Axes>,
        #   'TUNING_MAPS': <Figure size 1728x1080 with 88 Axes>},
        #  'maze2': {'SPIKES_MAPS': <Figure size 1728x864 with 71 Axes>,
        #   'TUNING_MAPS': <Figure size 1728x864 with 71 Axes>}}

        win, all_dock_display_items, all_nested_dock_area_widgets, all_nested_dock_area_widget_display_items = _build_docked_pf_2D_figures_widget(active_pf_2D_figures, should_nest_figures_on_filter=True, debug_print=False)

        win, all_dock_display_items, all_nested_dock_area_widgets, all_nested_dock_area_widget_display_items = _build_docked_pf_2D_figures_widget(active_pf_2D_figures, should_nest_figures_on_filter=True, debug_print=False)
        
        
    """
    min_width = 500
    min_height = 500
    if extant_dockAreaWidget is None:
        created_new_main_widget = True
        active_containing_dockAreaWidget, app = DockAreaWrapper._build_default_dockAreaWindow(title='active_pf_2D_figures', defer_show=False)
    else:
        created_new_main_widget = False
        active_containing_dockAreaWidget = extant_dockAreaWidget

    all_dock_display_items = {}
    all_item_widths_list = []
    all_item_heights_list = []

    if should_nest_figures_on_filter:
        all_nested_dock_area_widgets = {}
        all_nested_dock_area_widget_display_items = {}

        _last_dock_outer_nested_item = None
        for filter_name, a_figures_dict in active_pf_2D_figures.items():
            # For each filter, create a new NestedDockAreaWidget
            all_nested_dock_area_widgets[filter_name] = NestedDockAreaWidget()
            # Once done with a given filter, add its nested dockarea widget to the window
            if _last_dock_outer_nested_item is not None:
                #NOTE: to stack two dock widgets on top of each other, do area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
                dockAddLocationOpts = ['above', _last_dock_outer_nested_item] # position relative to the _last_dock_outer_nested_item for this figure
            else:
                dockAddLocationOpts = ['bottom'] #no previous dock for this filter, so use absolute positioning
            nested_out_widget_key = f'Nested Outer Widget: {filter_name}'
            if debug_print:
                print(f'nested_out_widget_key: {nested_out_widget_key}')
            _, dDisplayItem = active_containing_dockAreaWidget.add_display_dock(nested_out_widget_key, dockSize=(min_width, min_height), dockIsClosable=False, widget=all_nested_dock_area_widgets[filter_name], dockAddLocationOpts=dockAddLocationOpts)
            all_nested_dock_area_widget_display_items[filter_name] = dDisplayItem
            _last_dock_outer_nested_item = dDisplayItem

            ## Add the sub-items for this filter:
            _last_dock_item = None
            for a_figure_name, a_figure in a_figures_dict.items():
                # individual figures
                figure_key = f'{filter_name}_{a_figure_name}'
                if debug_print:
                    print(f'figure_key: {figure_key}')
                fig_window = a_figure.canvas.window()
                fig_geom = fig_window.window().geometry() # get the QTCore PyRect object
                fig_x, fig_y, fig_width, fig_height = fig_geom.getRect() # Note: dx & dy refer to width and height
                all_item_widths_list.append(fig_width)
                all_item_heights_list.append(fig_height)

                # Add the dock and keep the display item:
                if _last_dock_item is not None:
                    dockAddLocationOpts = ['above', _last_dock_item] # position relative to the _last_dock_item for this figure
                else:
                    dockAddLocationOpts = ['bottom'] #no previous dock for this filter, so use absolute positioning
                _, dDisplayItem = all_nested_dock_area_widgets[filter_name].add_display_dock(figure_key, dockSize=(fig_width, fig_height), dockIsClosable=False, widget=fig_window, dockAddLocationOpts=dockAddLocationOpts)
                dDisplayItem.setOrientation('horizontal') # want orientation of outer dockarea to be opposite of that of the inner one. # 'auto', 'horizontal', or 'vertical'.
                all_dock_display_items[figure_key] = dDisplayItem
                _last_dock_item = dDisplayItem

    else:
        ## Flat (non-nested)
        all_nested_dock_area_widgets = None
        all_nested_dock_area_widget_display_items = None
        
        for filter_name, a_figures_dict in active_pf_2D_figures.items():
            _last_dock_item = None
            for a_figure_name, a_figure in a_figures_dict.items():
                # individual figures
                figure_key = f'{filter_name}_{a_figure_name}'
                if debug_print:
                    print(f'figure_key: {figure_key}')
                fig_window = a_figure.canvas.window()
                fig_geom = fig_window.window().geometry() # get the QTCore PyRect object
                fig_x, fig_y, fig_width, fig_height = fig_geom.getRect() # Note: dx & dy refer to width and height
                all_item_widths_list.append(fig_width)
                all_item_heights_list.append(fig_height)
                
                # Add the dock and keep the display item:
                if _last_dock_item is not None:
                    #NOTE: to stack two dock widgets on top of each other, do area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
                    dockAddLocationOpts = ['above', _last_dock_item] # position relative to the _last_dock_item for this figure
                else:
                    dockAddLocationOpts = ['bottom'] #no previous dock for this filter, so use absolute positioning
                _, dDisplayItem = active_containing_dockAreaWidget.add_display_dock(figure_key, dockSize=(fig_width, fig_height), dockIsClosable=False, widget=fig_window, dockAddLocationOpts=dockAddLocationOpts)
                all_dock_display_items[figure_key] = dDisplayItem

                _last_dock_item = dDisplayItem

    # Resize window to largest figure size:
    if created_new_main_widget:
        # Only resize if we created this widget, otherwise don't change the size
        all_item_widths_list = np.array(all_item_widths_list)
        all_item_heights_list = np.array(all_item_heights_list)
        max_width = np.max(all_item_widths_list)
        max_height = np.max(all_item_heights_list)
        active_containing_dockAreaWidget.resize(max_width, max_height)
    
    return active_containing_dockAreaWidget, all_dock_display_items, all_nested_dock_area_widgets, all_nested_dock_area_widget_display_items
    

# ==================================================================================================================== #
# 2022-07-20                                                                                                           #
# ==================================================================================================================== #

def old_timesynchronized_plotter_testing():
    # # Test PfND_TimeDependent Class

    # ## Old TimeSynchronized*Plotter Testing

    # CELL ==================================================================================================================== #    t = curr_occupancy_plotter.active_time_dependent_placefields.last_t + 7 # add one second
    # with np.errstate(divide='ignore', invalid='ignore'):
    # active_time_dependent_placefields.update(t)
    print(f't: {t}')
    curr_occupancy_plotter.on_window_changed(0.0, t)
    curr_placefields_plotter.on_window_changed(0.0, t)


    # CELL ==================================================================================================================== #    from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image, pyqtplot_plot_image_array
    import time

    def _test_plot_curr_pf_result(curr_t, curr_ratemap, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ plots a single result at a given time.
        
        Creates the figures if needed, otherwise updates the existing ones.
        
        """
        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {}, 'placefields': {}} # make a new dictionary to hold the plot objects.

        # images = curr_ratemap.tuning_curves # (43, 63, 63)
        occupancy = curr_ratemap.occupancy
        # occupancy = curr_ratemap.curr_raw_occupancy_map

        imv = output_plots_dict.get('occupancy', {}).get('imv', None)
        if imv is None:
            # Otherwise build the plotter:
            occupancy_app, occupancy_win, imv = pyqtplot_plot_image(active_time_dependent_placefields2D.xbin, active_time_dependent_placefields2D.ybin, occupancy)
            output_plots_dict['occupancy'] = dict(zip(('app', 'win', 'imv'), (occupancy_app, occupancy_win, imv)))   
            occupancy_win.show()
        else:
            # Update the existing one:
            imv.setImage(occupancy, xvals=active_time_dependent_placefields2D.xbin)

        pg.QtGui.QApplication.processEvents() # call to ensure the occupancy gets updated before starting the placefield plots:
        
        img_item_array = output_plots_dict.get('placefields', {}).get('img_item_array', None)
        if img_item_array is None:
            # Create a new one:
            placefields_app, placefields_win, root_render_widget, plot_array, img_item_array, other_components_array = pyqtplot_plot_image_array(active_time_dependent_placefields2D.xbin, active_time_dependent_placefields2D.ybin,
                                                                                                                                            active_time_dependent_placefields2D.ratemap.normalized_tuning_curves, active_time_dependent_placefields2D.curr_raw_occupancy_map)#, 
            output_plots_dict['placefields'] = dict(zip(('app', 'win', 'root_render_widget', 'plot_array', 'img_item_array', 'other_components_array'), (placefields_app, placefields_win, root_render_widget, plot_array, img_item_array, other_components_array)))
            placefields_win.show()

        else:
            # Update the placefields plot if needed:
            images = curr_ratemap.tuning_curves # (43, 63, 63)
            for i, an_img_item in enumerate(img_item_array):
                image = np.squeeze(images[i,:,:])
                # Pre-filter the data:
                # image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
                if drop_below_threshold is not None:
                    image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy        
                # an_img_item.setImage(np.squeeze(images[i,:,:]))
                an_img_item.setImage(image)

        return output_plots_dict


    # CELL ==================================================================================================================== #
    def pre_build_iterative_results(num_iterations=50, t_list=[], ratemaps_list=[]):
        """ 
        build up historical data arrays:
        
        Usage:
            t_list, ratemaps_list = pre_build_iterative_results(num_iterations=50, t_list=t_list, ratemaps_list=ratemaps_list)
        """
        # t_list = []
        # ratemaps_list = []
        
        def _step_plot(time_step_seconds):
            t = active_time_dependent_placefields2D.last_t + time_step_seconds # add one second
            t_list.append(t)
            with np.errstate(divide='ignore', invalid='ignore'):
                active_time_dependent_placefields2D.update(t)
            # Loop through and update the plots:
            # Get flat list of images:
            curr_ratemap = active_time_dependent_placefields2D.ratemap
            # images = curr_ratemap.tuning_curves # (43, 63, 63)
            # images = active_time_dependent_placefields2D.ratemap.normalized_tuning_curves[0:40,:,:] # (43, 63, 63)
            # occupancy = curr_ratemap.occupancy
            ratemaps_list.append(curr_ratemap)
        #     for i, an_img_item in enumerate(img_item_array):
        #     # for i, a_plot in enumerate(plot_array):
        #         # image = np.squeeze(images[i,:,:])
        #         # Pre-filter the data:
        #         # image = np.array(image.copy()) / np.nanmax(image) # note scaling by maximum here!
        # #         if drop_below_threshold is not None:
        # #             image[np.where(occupancy < drop_below_threshold)] = np.nan # null out the occupancy        
        #         an_img_item.setImage(np.squeeze(images[i,:,:]))
        
        for i in np.arange(num_iterations):
            _step_plot(time_step_seconds=1.0)
        
        return t_list, ratemaps_list


    # Loop through the historically collected ratemaps and plot them:
    def _test_plot_historical_iterative_pf_results(t_list, ratemaps_list, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ Uses the previously built-up t_list and ratemaps_list (as computed by pre_build_iterative_results(...)) to plot the time-dependent results.
        requires:
        imv: a previously created single-image plotter:
        """
        num_historical_results = len(ratemaps_list)
        assert len(t_list) == len(ratemaps_list), f"len(t_list): {len(t_list)} needs to equal len(ratemaps_list): {len(ratemaps_list)}"
        
        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {},
                                'placefields': {}} # make a new dictionary to hold the plot objects.
            
        for i in np.arange(num_historical_results):
            curr_t = t_list[i]
            # Set up
            # print(f'curr_t: {curr_t}')
            curr_ratemap = ratemaps_list[i]
            output_plots_dict = _test_plot_curr_pf_result(curr_t, curr_ratemap, drop_below_threshold=drop_below_threshold, output_plots_dict=output_plots_dict)
        
            pg.QtGui.QApplication.processEvents()
            time.sleep(0.1) # Sleep for 0.5 seconds

        return output_plots_dict

    # Build the Historical Results:
    t_list, ratemaps_list = pre_build_iterative_results(num_iterations=50, t_list=t_list, ratemaps_list=ratemaps_list)
    # Plot the historical results:
    if output_plots_dict is None:
        output_plots_dict = {'occupancy': {}, 'placefields': {}}
    output_plots_dict = _test_plot_historical_iterative_pf_results(t_list, ratemaps_list, output_plots_dict=output_plots_dict)


    # CELL ==================================================================================================================== #
    # Compute the time-dependent ratemap info in real-time and plot them:
    def _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, t, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ 
        requires:
        imv: a previously created single-image plotter:
        """
        # Compute the updated placefields/occupancy for the time t:
        with np.errstate(divide='ignore', invalid='ignore'):
            active_time_dependent_placefields2D.update(t)
        # Update the plots:
        curr_t = active_time_dependent_placefields2D.last_t
        curr_ratemap = active_time_dependent_placefields2D.ratemap

        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {}, 'placefields': {}} # make a new dictionary to hold the plot objects.
            
        # Plot the results directly from the active_time_dependent_placefields2D
        output_plots_dict = _test_plot_curr_pf_result(curr_t, curr_ratemap, drop_below_threshold=drop_below_threshold, output_plots_dict=output_plots_dict)
        pg.QtGui.QApplication.processEvents()
        
        return output_plots_dict

    def _test_live_iterative_pf_results_plot(active_time_dependent_placefields2D, num_iterations=50, time_step_seconds=1.0, drop_below_threshold: float=0.0000001, output_plots_dict=None):
        """ performs num_iterations time steps of size time_step_seconds and plots the results. """
        for i in np.arange(num_iterations):
            t = active_time_dependent_placefields2D.last_t + time_step_seconds # add one second
            output_plots_dict = _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, t, drop_below_threshold=drop_below_threshold, output_plots_dict=output_plots_dict)
            time.sleep(0.1) # Sleep for 0.5 seconds


    # CELL ==================================================================================================================== #
    try:
        if output_plots_dict is None:
            output_plots_dict = {'occupancy': {}, 'placefields': {}}
    except NameError:
        output_plots_dict = {'occupancy': {}, 'placefields': {}}

    output_plots_dict = _test_live_iterative_pf_results_plot(active_time_dependent_placefields2D, num_iterations=50, time_step_seconds=1.0, output_plots_dict=output_plots_dict)


    # CELL ==================================================================================================================== #
    output_plots_dict = {'occupancy': {}, 'placefields': {}} # clear the output plots dict
    output_plots_dict = _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, spike_raster_window.spikes_window.active_time_window[1], output_plots_dict=output_plots_dict)


    # CELL ==================================================================================================================== #
    def _on_window_updated(window_start, window_end):
        # print(f'_on_window_updated(window_start: {window_start}, window_end: {window_end})')
        global output_plots_dict
        ## Update only version:
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     active_time_dependent_placefields2D.update(window_end) # advance the placefield display to the end of the window.
        ## Update and plot version:
        # t = window_end
        output_plots_dict = _test_step_live_iterative_pf_results_plot(active_time_dependent_placefields2D, window_end, output_plots_dict=output_plots_dict)
        
    # spike_raster_window.connect_additional_controlled_plotter(_on_window_updated)

    _on_window_updated(spike_raster_window.spikes_window.active_time_window[0], spike_raster_window.spikes_window.active_time_window[1])
    sync_connection = spike_raster_window.spike_raster_plt_2d.window_scrolled.connect(_on_window_updated) # connect the window_scrolled event to the _on_window_updated function


    # CELL ==================================================================================================================== #
    active_time_dependent_placefields2D.plot_occupancy()


    # CELL ==================================================================================================================== #
    # active_time_dependent_placefields2D.plot_ratemaps_2D(enable_spike_overlay=False) # Works
    active_time_dependent_placefields2D.plot_ratemaps_2D(enable_spike_overlay=True)


    # CELL ==================================================================================================================== #
    # t_list
    active_time_dependent_placefields2D.plot_ratemaps_2D(enable_saving_to_disk=False, enable_spike_overlay=False)


    # CELL ==================================================================================================================== #
    # ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(None, active_time_dependent_placefields2D, active_config_name)
    occupancy_fig, occupancy_ax = active_time_dependent_placefields2D.plot_occupancy(identifier_details_list=[])


    # CELL ==================================================================================================================== #
    i = 0
    while (i < len(t_list)):
        curr_t = t_list[i]
        # Set up
        print(f'curr_t: {curr_t}')
        curr_ratemap = ratemaps_list[i]
        # images = curr_ratemap.tuning_curves # (43, 63, 63)
        occupancy = curr_ratemap.occupancy
        # occupancy = curr_ratemap.curr_raw_occupancy_map
        imv.setImage(occupancy, xvals=active_time_dependent_placefields2D.xbin)
        i += 1
        pg.QtGui.QApplication.processEvents()
        
    print(f'done!')


    # CELL ==================================================================================================================== #
    # Timer Update Approach:
    timer = pg.QtCore.QTimer()
    i = 0
    def update():
        if (i < len(t_list)):
            curr_t = t_list[i]
            # Set up
            print(f'curr_t: {curr_t}')
            curr_ratemap = ratemaps_list[i]
            # images = curr_ratemap.tuning_curves # (43, 63, 63)
            occupancy = curr_ratemap.occupancy
            # occupancy = curr_ratemap.curr_raw_occupancy_map
            imv.setImage(occupancy, xvals=active_time_dependent_placefields2D.xbin)
            i += 1
        else:
            print(f'done!')
        # pw.plot(x, y, clear=True)

    timer.timeout.connect(update)


    # CELL ==================================================================================================================== #
    # timer.start(16)
    timer.start(500)


    # CELL ==================================================================================================================== #
    timer.stop()


    # CELL ==================================================================================================================== #
    t_list


    # CELL ==================================================================================================================== #
    # get properties from spike_raster_window:

    active_curve_plotter_3d = test_independent_vedo_raster_widget # use separate vedo plotter
    # active_curve_plotter_3d = spike_raster_window.spike_raster_plt_3d
    curr_computations_results = curr_active_pipeline.computation_results[active_config_name]


    # CELL ==================================================================================================================== #
    ## Spike Smoothed Moving Average Rate:
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves3D.Specific3DTimeCurves import Specific3DTimeCurvesHelper
    binned_spike_moving_average_rate_curve_datasource = Specific3DTimeCurvesHelper.add_unit_time_binned_spike_visualization_curves(curr_computations_results, active_curve_plotter_3d, spike_visualization_mode='mov_average')            


    # CELL ==================================================================================================================== #
    # Get current plot items:
    curr_plot3D_active_window_data = active_curve_plotter_3d.params.time_curves_datasource.get_updated_data_window(active_curve_plotter_3d.spikes_window.active_window_start_time, active_curve_plotter_3d.spikes_window.active_window_end_time) # get updated data for the active window from the datasource
    is_data_series_mode = active_curve_plotter_3d.params.time_curves_datasource.has_data_series_specs
    if is_data_series_mode:
        data_series_spaital_values_list = active_curve_plotter_3d.params.time_curves_datasource.data_series_specs.get_data_series_spatial_values(curr_plot3D_active_window_data)
        num_data_series = len(data_series_spaital_values_list)

    curr_data_series_index = 0
    curr_data_series_dict = data_series_spaital_values_list[curr_data_series_index]

    curr_plot_column_name = curr_data_series_dict.get('name', f'series[{curr_data_series_index}]') # get either the specified name or the generic 'series[i]' name otherwise
    curr_plot_name = active_curve_plotter_3d.params.time_curves_datasource.datasource_UIDs[curr_data_series_index]
    # points for the current plot:
    pts = np.column_stack([curr_data_series_dict['x'], curr_data_series_dict['y'], curr_data_series_dict['z']])
    pts


    # CELL ==================================================================================================================== #
    ## Add the new filled plot item:
    plot_args = ({'color_name':'white','line_width':0.5,'z_scaling_factor':1.0})
    _test_fill_plt = gl.GLLinePlotItem(pos=points, color=line_color, width=plot_args.setdefault('line_width',0.5), antialias=True)
    _test_fill_plt.scale(1.0, 1.0, plot_args.setdefault('z_scaling_factor',1.0)) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.            
    # plt.scale(1.0, 1.0, self.data_z_scaling_factor) # Scale the data_values_range to fit within the z_max_value. Shouldn't need to be adjusted so long as data doesn't change.
    active_curve_plotter_3d.ui.main_gl_widget.addItem(_test_fill_plt)


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.plots.keys()


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.render_epochs


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_datasource


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_enable_baseline_grid = True
    active_curve_plotter_3d.params.time_curves_baseline_grid_alpha = 0.9
    # add_3D_time_curves_baseline_grid_mesh


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.add_3D_time_curves_baseline_grid_mesh()


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.remove_3D_time_curves_baseline_grid_mesh()


    # CELL ==================================================================================================================== #
    list(active_curve_plotter_3d.plots.keys())


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.plots.time_curve_helpers


    # CELL ==================================================================================================================== #
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves3D.Render3DTimeCurvesBaseGridMixin import BaseGrid3DTimeCurvesHelper, Render3DTimeCurvesBaseGridMixin


    # CELL ==================================================================================================================== #
    BaseGrid3DTimeCurvesHelper.init_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d=active_curve_plotter_3d)


    # CELL ==================================================================================================================== #
    BaseGrid3DTimeCurvesHelper.add_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d=active_curve_plotter_3d)


    # CELL ==================================================================================================================== #
    BaseGrid3DTimeCurvesHelper.remove_3D_time_curves_baseline_grid_mesh(active_curve_plotter_3d=active_curve_plotter_3d)


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_main_alpha = 0.5
    active_curve_plotter_3d.update_3D_time_curves()


    # CELL ==================================================================================================================== #
    # Add default params if needed:
    # active_curve_plotter_3d.params


    # CELL ==================================================================================================================== #
    list(active_curve_plotter_3d.params.keys())


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.time_curves_z_baseline


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.plots


    # CELL ==================================================================================================================== #
    'time_curve_helpers' not in active_curve_plotter_3d.plots


    # CELL ==================================================================================================================== #
    'plots_grid_3dCurveBaselines_Grid' not in active_curve_plotter_3d.plots.time_curve_helpers


    # CELL ==================================================================================================================== #
    time_curves_z_baseline = 5.0 

    data_series_baseline
    # z_map_fn = lambda v_main: v_main + 5.0 # returns the un-transformed primary value

    5.0


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.params.axes_walls_z_height = 15.0


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d._update_axes_plane_graphics()


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.temporal_axis_length


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.temporal_zoom_factor # 2.6666666666666665


    # CELL ==================================================================================================================== #
    active_curve_plotter_3d.temporal_to_spatial(temporal_data=[1.0])

    # CELL ==================================================================================================================== #
    line_color = pg.mkColor(plot_args.setdefault('color_name', 'white'))
    line_color.setAlphaF(0.8)

# ==================================================================================================================== #
# Pre- 2022-07-11                                                                                                      #
# ==================================================================================================================== #

def process_by_good_placefields(session, active_config, active_placefields):
    """  Filters the session by the units in active_placefields that have good placefields and return an updated session. Also adds generated colors for each good unit to active_config """
    # Get the cell IDs that have a good place field mapping:
    good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
    print('good_placefield_neuronIDs: {}; ({} good)'.format(good_placefield_neuronIDs, len(good_placefield_neuronIDs)))

    ## Filter by neurons with good placefields only:
    good_placefields_session = session.get_by_id(good_placefield_neuronIDs) # Filter by good placefields only, and this fetch also ensures they're returned in the order of sorted ascending index ([ 2  3  5  7  9 12 18 21 22 23 26 27 29 34 38 45 48 53 57])

    pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    active_config.plotting_config.pf_sort_ind = pf_sort_ind
    active_config.plotting_config.pf_colors = pf_colors
    active_config.plotting_config.active_cells_colormap = pf_colormap
    active_config.plotting_config.active_cells_listed_colormap = ListedColormap(active_config.plotting_config.active_cells_colormap)
    
    return good_placefields_session, active_config, good_placefield_neuronIDs

def build_placefield_multiplotter(nfields, linear_plot_data=None):
    linear_plotter_indicies = np.arange(nfields)
    fixed_columns = 5
    needed_rows = int(np.ceil(nfields / fixed_columns))
    row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
    mp = pvqt.MultiPlotter(nrows=needed_rows, ncols=fixed_columns, show=False, title='Muliplotter', toolbar=False, menu_bar=False, editor=False)
    print('linear_plotter_indicies: {}\n row_column_indicies: {}\n'.format(linear_plotter_indicies, row_column_indicies))
    # mp[0, 0].add_mesh(pv.Sphere())
    # mp[0, 1].add_mesh(pv.Cylinder())
    # mp[1, 0].add_mesh(pv.Cube())
    # mp[1, 1].add_mesh(pv.Cone())
    for a_linear_index in linear_plotter_indicies:
        print('a_linear_index: {}, row_column_indicies[0][a_linear_index]: {}, row_column_indicies[1][a_linear_index]: {}'.format(a_linear_index, row_column_indicies[0][a_linear_index], row_column_indicies[1][a_linear_index]))
        curr_row = row_column_indicies[0][a_linear_index]
        curr_col = row_column_indicies[1][a_linear_index]
        if linear_plot_data is None:
            mp[curr_row, curr_col].add_mesh(pv.Sphere())
        else:
            mp[curr_row, curr_col].add_mesh(linear_plot_data[a_linear_index], name='maze_bg', color="black", render=False)
            # mp[a_row_column_index[0], a_row_column_index[1]].add_mesh(pv.Sphere())
    return mp, linear_plotter_indicies, row_column_indicies

#TODO: Works, but need to convert into the computation function format or find a new place to put it. It operates on the entire pipeline while currently computation functions are limited to operating on one stage at a time.
def _perform_PBE_stats(active_pipeline, debug_print = False):
    """ # Analyze PBEs by looping through the filtered epochs:
        This whole implementation seems silly and inefficient        
        Can't I use .agg(['count', 'mean']) or something? 
        
        
    Usage:
        from PendingNotebookCode import _perform_PBE_stats
        pbe_analyses_result_df, [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] = _perform_PBE_stats(curr_active_pipeline, debug_print=False) # all_epochs_n_pbes: [206, 31, 237], all_epochs_mean_pbe_durations: [0.2209951456310722, 0.23900000000001073, 0.22335021097046923], all_epochs_cummulative_pbe_durations: [45.52500000000087, 7.409000000000333, 52.934000000001205], all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, 1910.1600048116618]
        pbe_analyses_result_df

    """
    all_epochs_labels = []
    all_epochs_total_durations = []
    all_epochs_n_pbes = []
    all_epochs_pbe_duration_lists = []
    all_epochs_cummulative_pbe_durations = []
    all_epochs_mean_pbe_durations = []
    all_epochs_full_pbe_spiketrain_lists = []
    all_epochs_pbe_num_spikes_lists = []
    all_epochs_intra_pbe_interval_lists = []
    
    for (name, filtered_sess) in active_pipeline.filtered_sessions.items():
        # interested in analyzing both the filtered_sess.pbe and the filtered_sess.spikes_df (as they relate to the PBEs)
        all_epochs_labels.append(name)
        curr_named_time_range = active_pipeline.sess.epochs.get_named_timerange(name) # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
        
        if not np.isscalar(curr_named_time_range.duration):
            # for 'maze' key, the total duration is being set to array([], dtype=float64) for some reason. all_epochs_total_durations: [1716.8933641185379, 193.26664069312392, array([], dtype=float64)]
            curr_named_time_range = NamedTimerange(name='maze', start_end_times=[active_pipeline.sess.epochs['maze1'][0], active_pipeline.sess.epochs['maze2'][1]])
        
        curr_epoch_duration = curr_named_time_range.duration
        all_epochs_total_durations.append(curr_epoch_duration) # TODO: this should be in seconds (or at least the same units as the PBE durations)... actually this might be right.
        # Computes the intervals between each PBE:
        curr_intra_pbe_intervals = filtered_sess.pbe.starts[1:] - filtered_sess.pbe.stops[:-1]
        all_epochs_intra_pbe_interval_lists.append(curr_intra_pbe_intervals)
        all_epochs_n_pbes.append(filtered_sess.pbe.n_epochs)
        all_epochs_pbe_duration_lists.append(filtered_sess.pbe.durations)
        all_epochs_cummulative_pbe_durations.append(np.sum(filtered_sess.pbe.durations))
        all_epochs_mean_pbe_durations.append(np.nanmean(filtered_sess.pbe.durations))
        # filtered_sess.spikes_df.PBE_id
        curr_pbe_only_spikes_df = filtered_sess.spikes_df[filtered_sess.spikes_df.PBE_id > -1].copy()
        unique_PBE_ids = np.unique(curr_pbe_only_spikes_df['PBE_id'])
        flat_PBE_ids = [int(id) for id in unique_PBE_ids]
        num_unique_PBE_ids = len(flat_PBE_ids)
        # groups the spikes_df by PBEs:
        curr_pbe_grouped_spikes_df = curr_pbe_only_spikes_df.groupby(['PBE_id'])
        curr_spiketrains = list()
        curr_PBE_spiketrain_num_spikes = list()
        for i in np.arange(num_unique_PBE_ids):
            curr_PBE_id = flat_PBE_ids[i] # actual cell ID
            #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
            curr_PBE_dataframe = curr_pbe_grouped_spikes_df.get_group(curr_PBE_id)
            curr_PBE_num_spikes = np.shape(curr_PBE_dataframe)[0] # the number of spikes in this PBE
            curr_PBE_spiketrain_num_spikes.append(curr_PBE_num_spikes)
            curr_spiketrains.append(curr_PBE_dataframe['t'].to_numpy())

        curr_PBE_spiketrain_num_spikes = np.array(curr_PBE_spiketrain_num_spikes)
        all_epochs_pbe_num_spikes_lists.append(curr_PBE_spiketrain_num_spikes)
        curr_spiketrains = np.array(curr_spiketrains, dtype='object')
        all_epochs_full_pbe_spiketrain_lists.append(curr_spiketrains)
        if debug_print:
            print(f'name: {name}, filtered_sess.pbe: {filtered_sess.pbe}')

    if debug_print:
        print(f'all_epochs_n_pbes: {all_epochs_n_pbes}, all_epochs_mean_pbe_durations: {all_epochs_mean_pbe_durations}, all_epochs_cummulative_pbe_durations: {all_epochs_cummulative_pbe_durations}, all_epochs_total_durations: {all_epochs_total_durations}')
        # all_epochs_n_pbes: [3152, 561, 1847, 832, 4566], all_epochs_mean_pbe_durations: [0.19560881979695527, 0.22129233511594312, 0.19185056848946497, 0.2333112980769119, 0.1987152869032212]

    all_epochs_pbe_occurance_rate = [(float(all_epochs_total_durations[i]) / float(all_epochs_n_pbes[i])) for i in np.arange(len(all_epochs_n_pbes))]
    all_epochs_pbe_percent_duration = [(float(all_epochs_total_durations[i]) / float(all_epochs_cummulative_pbe_durations[i])) for i in np.arange(len(all_epochs_n_pbes))]    
    all_epoch_mean_num_pbe_spikes = [np.nanmean(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [3151, 561, 1847, 831, 4563]
    all_epoch_std_num_pbe_spikes = [np.nanstd(pbe_spike_counts) for pbe_spike_counts in all_epochs_pbe_num_spikes_lists] # [11.638970035733648, 15.013817202645336, 15.5123897729991, 15.113395025612247, 11.473087401691878]
    # [20.429704855601397, 27.338680926916222, 23.748781808337846, 25.673886883273166, 20.38614946307254]
    # Build the final output result dataframe:
    pbe_analyses_result_df = pd.DataFrame({'n_pbes':all_epochs_n_pbes, 'mean_pbe_durations': all_epochs_mean_pbe_durations, 'cummulative_pbe_durations':all_epochs_cummulative_pbe_durations, 'epoch_total_duration':all_epochs_total_durations,
                'pbe_occurance_rate':all_epochs_pbe_occurance_rate, 'pbe_percent_duration':all_epochs_pbe_percent_duration,
                'mean_num_pbe_spikes':all_epoch_mean_num_pbe_spikes, 'stddev_num_pbe_spikes':all_epoch_std_num_pbe_spikes}, index=all_epochs_labels)
    # temporary: this isn't how the returns work for other computation functions:
    all_epochs_info = [all_epochs_full_pbe_spiketrain_lists, all_epochs_pbe_num_spikes_lists, all_epochs_intra_pbe_interval_lists] # list version
    # all_epochs_info = {'all_epochs_full_pbe_spiketrain_lists':all_epochs_full_pbe_spiketrain_lists, 'all_epochs_pbe_num_spikes_lists':all_epochs_pbe_num_spikes_lists, 'all_epochs_intra_pbe_interval_lists':all_epochs_intra_pbe_interval_lists} # dict version
    return pbe_analyses_result_df, all_epochs_info


# -------------------------- 2022-06-22 Notebook 93 -------------------------- #
import matplotlib.pyplot as plt

def spike_count_and_firing_rate_normalizations(pho_custom_decoder, enable_plots=True):
    """ Computes several different normalizations of binned firing rate and spike counts, optionally plotting them. 
    
    Usage:
        pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
        enable_plots = True
        unit_specific_time_binned_outputs = spike_count_and_firing_rate_normalizations(pho_custom_decoder, enable_plots=enable_plots)
        spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = unit_specific_time_binned_outputs # unwrap the output tuple:
        
        
    TESTING CODE:
    
    pho_custom_decoder = curr_kdiba_pipeline.computation_results['maze1'].computed_data['pf2D_Decoder']
    enable_plots = True

    print(f'most_likely_positions: {np.shape(pho_custom_decoder.most_likely_positions)}') # most_likely_positions: (3434, 2)
    unit_specific_time_binned_outputs = spike_count_and_firing_rate_normalizations(pho_custom_decoder, enable_plots=enable_plots)
    spike_proportion_global_fr_normalized, firing_rate, firing_rate_global_fr_normalized = unit_specific_time_binned_outputs # unwrap the output tuple:

    # pho_custom_decoder.unit_specific_time_binned_spike_counts.shape # (64, 1717)
    unit_specific_binned_spike_count_mean = np.nanmean(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
    unit_specific_binned_spike_count_var = np.nanvar(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
    unit_specific_binned_spike_count_median = np.nanmedian(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)

    unit_specific_binned_spike_count_mean
    unit_specific_binned_spike_count_median
    # unit_specific_binned_spike_count_mean.shape # (64, )

    """
    # produces a fraction which indicates which proportion of the window's firing belonged to each unit (accounts for global changes in firing rate (each window is scaled by the toial spikes of all cells in that window)
    unit_specific_time_binned_spike_proportion_global_fr_normalized = pho_custom_decoder.unit_specific_time_binned_spike_counts / pho_custom_decoder.total_spike_counts_per_window
    if enable_plots:
        plt.figure(num=5)
        plt.imshow(unit_specific_time_binned_spike_proportion_global_fr_normalized, cmap='turbo', aspect='auto')
        plt.title('Unit Specific Proportion of Window Spikes')
        plt.xlabel('Binned Time Window')
        plt.ylabel('Neuron Proportion Activity')

    # print(pho_custom_decoder.time_window_edges_binning_info.step)
    # print(f'pho_custom_decoder: {pho_custom_decoder}')
    # np.shape(pho_custom_decoder.F) # (1856, 64)

    unit_specific_time_binned_firing_rate = pho_custom_decoder.unit_specific_time_binned_spike_counts / pho_custom_decoder.time_window_edges_binning_info.step
    # print(unit_specific_time_binned_firing_rate)
    if enable_plots:
        plt.figure(num=6)
        plt.imshow(unit_specific_time_binned_firing_rate, cmap='turbo', aspect='auto')
        plt.title('Unit Specific Binned Firing Rates')
        plt.xlabel('Binned Time Window')
        plt.ylabel('Neuron Firing Rate')


    # produces a unit firing rate for each window that accounts for global changes in firing rate (each window is scaled by the firing rate of all cells in that window
    unit_specific_time_binned_firing_rate_global_fr_normalized = unit_specific_time_binned_spike_proportion_global_fr_normalized / pho_custom_decoder.time_window_edges_binning_info.step
    if enable_plots:
        plt.figure(num=7)
        plt.imshow(unit_specific_time_binned_firing_rate_global_fr_normalized, cmap='turbo', aspect='auto')
        plt.title('Unit Specific Binned Firing Rates (Global Normalized)')
        plt.xlabel('Binned Time Window')
        plt.ylabel('Neuron Proportion Firing Rate')
        
        
    # Special:
    # pho_custom_decoder.unit_specific_time_binned_spike_counts
    # unit_specific_binned_spike_count_mean = np.nanmean(pho_custom_decoder.unit_specific_time_binned_spike_counts, axis=1)
    

    # Return the computed values, leaving the original data unchanged.
    return unit_specific_time_binned_spike_proportion_global_fr_normalized, unit_specific_time_binned_firing_rate, unit_specific_time_binned_firing_rate_global_fr_normalized


