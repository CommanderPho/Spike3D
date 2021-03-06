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
from neuropy.core import Laps
from neuropy.core.epoch import NamedTimerange
from neuropy.analyses.laps import estimate_laps, compute_laps_spike_indicies
from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import plot_laps_2d

should_force_recompute_placefields = True
should_display_2D_plots = True



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
# 2022-07-11                                                                                                           #
# ==================================================================================================================== #
def _build_programmatic_display_function_testing_pdf_metadata(curr_active_pipeline, filter_name, out_path=None, debug_print=False):
    """ Builds the PDF metadata generating function from the passed info
    
        curr_active_pipeline: Needed for curr_active_pipeline.sess.get_session_description(curr_active_pipeline.session_data_type)
        filter_name: a name like 'maze1'
        out_path: an optional Path to use instead of generating a new one
        
    Returns:
        a function that takes one argument, the display function name, and returns the PDF metadata
        
    Usage:
        _build_pdf_pages_output_info, out_parent_path = _build_programmatic_display_function_testing_pdf_metadata(curr_active_pipeline, filter_name=active_config_name, out_path=None)
        
        curr_display_function_name = '_display_1d_placefield_validations'
        built_pdf_metadata, curr_pdf_save_path = _build_pdf_pages_output_info(curr_display_function_name)
        with backend_pdf.PdfPages(curr_pdf_save_path, keep_empty=False, metadata=built_pdf_metadata) as pdf:
            # plt.ioff() # disable displaying the plots inline in the Jupyter-lab notebook. NOTE: does not work in Jupyter-Lab, figures still show
            plots = curr_active_pipeline.display(curr_display_function_name, active_config_name) # works, but generates a TON of plots!
            # plt.ion()
            for fig_idx, a_fig in enumerate(plots):
                # print(f'saving fig: {fig_idx+1}/{len(plots)}')
                pdf.savefig(a_fig)
                # pdf.savefig(a_fig, transparent=True)
            # When no figure is specified the current figure is saved
            # pdf.savefig()

        
    """
    if out_path is None:   
        out_path = Path(r'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Screenshots\ProgrammaticDisplayFunctionTesting\2022-07-11')
    else:
        out_path = Path(out_path) # make sure it's a Path
    out_path.mkdir(exist_ok=True)

    session_descriptor_string = curr_active_pipeline.sess.get_session_description(curr_active_pipeline.session_data_type) # 'sess_kdiba_2006-6-07_11-26-53'
    pho_pdf_metadata = {'Creator': 'Spike3D - TestNeuroPyPipeline116', 'Author': 'Pho Hale', 'Title': session_descriptor_string, 'Subject': '', 'Keywords': [session_descriptor_string]}
    if debug_print:
        print(f'filter_name: {filter_name}')

    def _build_pdf_pages_output_info(display_function_name):
        """ 
        Implicitly captures:
            programmatic_display_fcn_out_path
            session_descriptor_string
            pho_pdf_metadata
            filter_name
        """
        built_pdf_metadata = pho_pdf_metadata.copy()
        context_tuple = [session_descriptor_string, filter_name, display_function_name]
        built_pdf_metadata['Title'] = '_'.join(context_tuple)
        built_pdf_metadata['Subject'] = display_function_name
        built_pdf_metadata['Keywords'] = ' | '.join(context_tuple)
        curr_pdf_save_path = out_path.joinpath(('_'.join(context_tuple) + '.pdf'))
        return built_pdf_metadata, curr_pdf_save_path
    
    return _build_pdf_pages_output_info, out_path



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


