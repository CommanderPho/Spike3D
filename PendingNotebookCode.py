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
from neuropy.core.neuron_identities import PlotStringBrevityModeEnum # for display_all_pf_2D_pyqtgraph_binned_image_rendering
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer

## Laps Stuff:
from neuropy.core.epoch import NamedTimerange

should_force_recompute_placefields = True
should_display_2D_plots = True
_debug_print = False

# ==================================================================================================================== #
# 2022-12-18 - Added Standardization of Position bins between short and long                                           #
# ==================================================================================================================== #

from neuropy.analyses.placefields import PfND # for re-binning pf1D
from pyphoplacecellanalysis.General.Mixins.CrossComputationComparisonHelpers import _compare_computation_results


### Piso-based interval overlap removal
# ## Build non-overlapping intervals with piso. Unsure of the computation efficiency, but the ouptuts are correct.
# import piso
# piso.register_accessors()

# print(f'pre: {active_filter_epochs.shape[0]}')
# valid_intervals = pd.arrays.IntervalArray.from_arrays(left=active_filter_epochs.start.values, right=active_filter_epochs.end.values).piso.symmetric_difference()
# valid_active_filter_epochs = np.vstack([valid_intervals.left.values.T, valid_intervals.right.values.T]).T
# print(f'post: {valid_active_filter_epochs.shape[0]}') # (37, 2)

# active_filter_epochs = valid_active_filter_epochs


def interleave(list1, list2):
    """ Chat-GPT """
    return [x for pair in zip(list1, list2) for x in pair]


import itertools
def interleave(list1, list2):
    """ human solution """
    return [x for x in itertools.chain.from_iterable(itertools.izip_longest(list1,list2)) if x]




def _get_common_cell_pf_results(long_neuron_ids, short_neuron_ids):
    ## get shared neuron info:
    # this must be done after we rebuild the short_pf1D bins (if we need to) so they continue to match:
    pf_neurons_diff = _compare_computation_results(long_neuron_ids, short_neuron_ids)

    shared_aclus = pf_neurons_diff.intersection #.shape (56,)
    print(f'shared_aclus: {shared_aclus}.\t np.shape: {np.shape(shared_aclus)}')
    # curr_any_context_neurons = pf_neurons_diff.either
    long_only_aclus = pf_neurons_diff.lhs_only
    short_only_aclus = pf_neurons_diff.rhs_only
    print(f'long_only_aclus: {long_only_aclus}.\t np.shape: {np.shape(long_only_aclus)}')
    print(f'short_only_aclus: {short_only_aclus}.\t np.shape: {np.shape(short_only_aclus)}')

    ## Get the normalized_tuning_curves only for the shared aclus (that are common across (long/short/global):
    long_is_included = np.isin(long_neuron_ids, shared_aclus)  #.shape # (104, 63)
    long_incl_aclus = np.array(long_neuron_ids)[long_is_included] #.shape # (98,)
    long_incl_curves = long_pf1D.ratemap.normalized_tuning_curves[long_is_included]  #.shape # (98, 63)
    assert long_incl_aclus.shape[0] == long_incl_curves.shape[0] # (98,) == (98, 63)

    short_is_included = np.isin(short_neuron_ids, shared_aclus)
    short_incl_aclus = np.array(short_neuron_ids)[short_is_included] #.shape (98,)
    short_incl_curves = short_pf1D.ratemap.normalized_tuning_curves[short_is_included]  #.shape # (98, 40)
    assert short_incl_aclus.shape[0] == short_incl_curves.shape[0] # (98,) == (98, 63)
    # assert short_incl_curves.shape[1] == long_incl_curves.shape[1] # short and long should have the same bins

    global_is_included = np.isin(global_neuron_ids, shared_aclus)
    global_incl_aclus = np.array(global_neuron_ids)[global_is_included] #.shape (98,)
    global_incl_curves = global_pf1D.ratemap.normalized_tuning_curves[global_is_included]  #.shape # (98, 63)
    assert global_incl_aclus.shape[0] == global_incl_curves.shape[0] # (98,) == (98, 63)
    assert global_incl_curves.shape[1] == long_incl_curves.shape[1] # global and long should have the same bins

    assert np.alltrue(np.isin(long_incl_aclus, short_incl_aclus))
    assert np.alltrue(np.isin(long_incl_aclus, global_incl_aclus))
    return 


# ==================================================================================================================== #
# 2022-12-15 Importing from TestNeuropyPipeline241                                                                     #
# ==================================================================================================================== #

def _update_nearest_decoded_most_likely_position_callback(start_t, end_t):
    """ Only uses end_t
    Implicitly captures: ipspikesDataExplorer, _get_nearest_decoded_most_likely_position_callback
    
    Usage:
        _update_nearest_decoded_most_likely_position_callback(0.0, ipspikesDataExplorer.t[0])
        _conn = ipspikesDataExplorer.sigOnUpdateMeshes.connect(_update_nearest_decoded_most_likely_position_callback)

    """
    def _get_nearest_decoded_most_likely_position_callback(t):
        """ A callback that when passed a visualization timestamp (the current time to render) returns the most likely predicted position provided by the active_two_step_decoder
        Implicitly captures:
            active_one_step_decoder, active_two_step_decoder
        Usage:
            _get_nearest_decoded_most_likely_position_callback(9000.1)
        """
        active_time_window_variable = active_one_step_decoder.time_window_centers # get time window centers (n_time_window_centers,) # (4060,)
        active_most_likely_positions = active_one_step_decoder.most_likely_positions.T # (4060, 2) NOTE: the most_likely_positions for the active_one_step_decoder are tranposed compared to the active_two_step_decoder
        # active_most_likely_positions = active_two_step_decoder.most_likely_positions # (2, 4060)
        assert np.shape(active_time_window_variable)[0] == np.shape(active_most_likely_positions)[1], f"timestamps and num positions must be the same but np.shape(active_time_window_variable): {np.shape(active_time_window_variable)} and np.shape(active_most_likely_positions): {np.shape(active_most_likely_positions)}!"
        last_window_index = np.searchsorted(active_time_window_variable, t, side='left') # side='left' ensures that no future values (later than 't') are ever returned
        # TODO: CORRECTNESS: why is it returning an index that corresponds to a time later than the current time?
        # for current time t=9000.0
        #     last_window_index: 1577
        #     last_window_time: 9000.5023
        # EH: close enough
        last_window_time = active_time_window_variable[last_window_index] # If there is no suitable index, return either 0 or N (where N is the length of `a`).
        displayed_time_offset = t - last_window_time # negative value if the window time being displayed is in the future
        if _debug_print:
            print(f'for current time t={t}\n\tlast_window_index: {last_window_index}\n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}')
        return (last_window_time, *list(np.squeeze(active_most_likely_positions[:, last_window_index]).copy()))

    t = end_t # the t under consideration should always be the end_t. This is written this way just for compatibility with the ipspikesDataExplorer.sigOnUpdateMeshes (float, float) signature
    curr_t, curr_x, curr_y = _get_nearest_decoded_most_likely_position_callback(t)
    curr_debug_point = [curr_x, curr_y, ipspikesDataExplorer.z_fixed[-1]]
    if _debug_print:
        print(f'tcurr_debug_point: {curr_debug_point}') # \n\tlast_window_time: {last_window_time}\n\tdisplayed_time_offset: {displayed_time_offset}
    ipspikesDataExplorer.perform_plot_location_point('debug_point_plot', curr_debug_point, color='r', render=True)
    return curr_debug_point


# ==================================================================================================================== #
# 2022-12-15 Finishing Up Surprise                                                                                     #
# ==================================================================================================================== #

def find_epoch_names(curr_active_pipeline):
    """ Returns the [long, short, global] epoch names. They must exist.
    Usage:
        from PendingNotebookCode import find_epoch_names
        long_epoch_name, short_epoch_name, global_epoch_name = find_epoch_names(curr_active_pipeline)
        long_results = curr_active_pipeline.computation_results[long_epoch_name]['computed_data']
        short_results = curr_active_pipeline.computation_results[short_epoch_name]['computed_data']
        global_results = curr_active_pipeline.computation_results[global_epoch_name]['computed_data']

    """
    include_whitelist = curr_active_pipeline.active_completed_computation_result_names # ['maze', 'sprinkle']
    long_epoch_name = include_whitelist[0] # 'maze1_PYR'
    short_epoch_name = include_whitelist[1] # 'maze2_PYR'
    global_epoch_name = include_whitelist[-1] # 'maze_PYR'
    return long_epoch_name, short_epoch_name, global_epoch_name


# ==================================================================================================================== #
# 2022-12-14 Batch Surprise Recomputation                                                                              #
# ==================================================================================================================== #

from numpy import inf # for _normalize_flat_relative_entropy_infs
from sklearn.preprocessing import minmax_scale # for _normalize_flat_relative_entropy_infs
from matplotlib.collections import BrokenBarHCollection # for add_epochs

def build_epoch_label(xy, text, ax, **labels_kwargs):
    """ places a text label for the epoch at the top, just inside of it """
    if labels_kwargs is None:
        labels_kwargs = {}
    labels_y_offset = labels_kwargs.pop('y_offset', -0.05)
    # y = xy[1]
    y = xy[1] + labels_y_offset  # shift y-value for label so that it's below the artist
    return ax.text(xy[0], y, text, **({'ha': 'center', 'va': 'top', 'family': 'sans-serif', 'size': 14} | labels_kwargs)) # va="top" places it inside the box if it's aligned to the top

def add_epochs(epoch_obj, curr_ax, facecolor=('green','red'), edgecolors=("black",), alpha=0.25, labels_kwargs=None, defer_render=False, debug_print=False):
    """ plots epoch rectangles on the existing matplotlib axis
    2022-12-14

    Usage:
        epochs_collection, epoch_labels = add_epochs(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)

    Full Usage Examples:

    ## Example 1:
        active_filter_epochs = curr_active_pipeline.sess.replay
        active_filter_epochs

        if not 'stop' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
            
        if not 'label' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

        active_filter_epoch_obj = Epoch(active_filter_epochs)
        active_filter_epoch_obj


        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_surprise_across_all_positions)
        ax.set_ylabel('Relative Entropy across all positions')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = add_epochs(curr_active_pipeline.sess.epochs, ax, facecolor=('red','cyan'), alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
        laps_epochs_collection, laps_epoch_labels = add_epochs(curr_active_pipeline.sess.laps.as_epoch_obj(), ax, facecolor='red', edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False)
        replays_epochs_collection, replays_epoch_labels = add_epochs(active_filter_epoch_obj, ax, facecolor='orange', edgecolors=None, labels_kwargs=None, defer_render=False, debug_print=False)
        fig.show()


    ## Example 2:

        # Show basic relative entropy vs. time plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_relative_entropy_results)
        ax.set_ylabel('Relative Entropy')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = add_epochs(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)
        fig.show()

    """        
    epoch_tuples = [(start_t, width_duration) for start_t, width_duration in zip(epoch_obj.starts, epoch_obj.durations)] # [(0.0, 1211.5580800310709), (1211.5580800310709, 882.3397767931456)]
    epoch_mid_t = [a_tuple[0]+(0.5*a_tuple[1]) for a_tuple in epoch_tuples] # used for labels

    curr_span_ymin = curr_ax.get_ylim()[0]
    curr_span_ymax = curr_ax.get_ylim()[1]
    curr_span_height = curr_span_ymax-curr_span_ymin
    # xrange: list of (float, float) The sequence of (left-edge-position, width) pairs for each bar.
    # yrange: (lower-edge, height) 
    epochs_collection = BrokenBarHCollection(xranges=epoch_tuples, yrange=(curr_span_ymin, curr_span_height), facecolor=facecolor, alpha=alpha, edgecolors=edgecolors, linewidths=(1,)) # , offset_transform=curr_ax.transData
    if debug_print:
        print(f'(curr_span_ymin, curr_span_ymax): ({curr_span_ymin}, {curr_span_ymax}), epoch_tuples: {epoch_tuples}')
    curr_ax.add_collection(epochs_collection)
    if labels_kwargs is not None:
        epoch_labels = [build_epoch_label((a_mid_t, curr_span_ymax), a_label, curr_ax, **labels_kwargs) for a_label, a_mid_t in zip(epoch_obj.labels, epoch_mid_t)]
    else:
        epoch_labels = None
    if not defer_render:
        curr_ax.get_figure().canvas.draw()
    return epochs_collection, epoch_labels

def _normalize_flat_relative_entropy_infs(flat_relative_entropy_results):
    """ Replace np.inf with a maximally high value.
    2022-12-14 WIP

    Usage:
        normalized_flat_relative_entropy_results = _normalize_flat_relative_entropy_infs(flat_relative_entropy_results)
    """

    # Replace np.inf with a maximally high value.
    inf_value_mask = np.isinf(flat_relative_entropy_results) # all the infinte values

    normalized_flat_relative_entropy_results = flat_relative_entropy_results.copy()
    normalized_flat_relative_entropy_results[normalized_flat_relative_entropy_results == inf] = 0  # zero out the infinite values for normalization to the feature range (-1, 1)
    normalized_flat_relative_entropy_results = minmax_scale(normalized_flat_relative_entropy_results, feature_range=(-1, 1)) # normalize to the feature_range (-1, 1)

    # Restore the infinite values at the specified value:
    # normalized_flat_relative_entropy_results[inf_value_mask] = 0.0
    return normalized_flat_relative_entropy_results


# ==================================================================================================================== #
# 2022-12-13 Misc                                                                                                      #
# ==================================================================================================================== #

def process_session_plots(curr_active_pipeline, active_config_name, debug_print=False):
    """ Unwrap single config 
    UNUSED AND UNTESTED

    Usage:

        from PendingNotebookCode import process_session_plots

        # active_config_name = 'maze1'
        # active_config_name = 'maze2'
        # active_config_name = 'maze'
        # active_config_name = 'sprinkle'

        # active_config_name = 'maze_PYR'

        # active_config_name = 'maze1_rippleOnly'
        # active_config_name = 'maze2_rippleOnly'

        # active_config_name = curr_active_pipeline.active_config_names[0] # get the first name by default
        active_config_name = curr_active_pipeline.active_config_names[-1] # get the last name

        active_identifying_filtered_session_ctx, programmatic_display_function_testing_output_parent_path = process_session_plots(curr_active_pipeline, active_config_name)

    """
    print(f'active_config_name: {active_config_name}')

    active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

    ## Add the filter to the active context
    # active_identifying_filtered_session_ctx = active_identifying_session_ctx.adding_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
    active_identifying_filtered_session_ctx = curr_active_pipeline.filtered_contexts[active_config_name] # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

    # Get relevant variables:
    # curr_active_pipeline is set above, and usable here
    sess = curr_active_pipeline.filtered_sessions[active_config_name]

    active_computation_results = curr_active_pipeline.computation_results[active_config_name]
    active_computed_data = curr_active_pipeline.computation_results[active_config_name].computed_data
    active_computation_config = curr_active_pipeline.computation_results[active_config_name].computation_config
    active_computation_errors = curr_active_pipeline.computation_results[active_config_name].accumulated_errors
    print(f'active_computed_data.keys(): {list(active_computed_data.keys())}')
    print(f'active_computation_errors: {active_computation_errors}')
    active_pf_1D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf1D']
    active_pf_2D = curr_active_pipeline.computation_results[active_config_name].computed_data['pf2D']
    active_pf_1D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_dt', None)
    active_pf_2D_dt = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_dt', None)
    active_firing_rate_trends = curr_active_pipeline.computation_results[active_config_name].computed_data.get('firing_rate_trends', None)
    active_one_step_decoder_2D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None) # BayesianPlacemapPositionDecoder
    active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None) 
    active_one_step_decoder_1D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_Decoder', None) # BayesianPlacemapPositionDecoder
    active_two_step_decoder_1D = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf1D_TwoStepDecoder', None)
    active_extended_stats = curr_active_pipeline.computation_results[active_config_name].computed_data.get('extended_stats', None)
    active_eloy_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('EloyAnalysis', None)
    active_simpler_pf_densities_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('SimplerNeuronMeetingThresholdFiringAnalysis', None)
    active_ratemap_peaks_analysis = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', None)
    active_peak_prominence_2d_results = curr_active_pipeline.computation_results[active_config_name].computed_data.get('RatemapPeaksAnalysis', {}).get('PeakProminence2D', None)
    active_measured_positions = curr_active_pipeline.computation_results[active_config_name].sess.position.to_dataframe()
    curr_spikes_df = sess.spikes_df

    curr_active_config = curr_active_pipeline.active_configs[active_config_name]
    curr_active_display_config = curr_active_config.plotting_config

    active_display_output = curr_active_pipeline.display_output[active_identifying_filtered_session_ctx]
    print(f'active_display_output: {active_display_output}')

    # Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18)
    display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_session_ctx, enable_gui=False, debug_print=True) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
    master_dock_win = display_output['master_dock_win']
    app = display_output['app']
    out_items = display_output['out_items']

    def _get_curr_figure_format_config():
        """ Aims to fetch the current figure_format_config and context from the figure_format_config widget:    
        Implicitly captures: `out_items`, `active_config_name`, `active_identifying_filtered_session_ctx` 
        """
        ## Get the figure_format_config from the figure_format_config widget:
        # Fetch the context from the GUI:
        _curr_gui_session_ctx, _curr_gui_out_display_items = out_items[active_config_name]
        _curr_gui_figure_format_config_widget = _curr_gui_out_display_items[active_identifying_filtered_session_ctx.adding_context('display_fn', display_fn_name='figure_format_config_widget')] # [0] is seemingly not needed to unpack the tuple
        if _curr_gui_figure_format_config_widget is not None:
            # has GUI for config
            figure_format_config = _curr_gui_figure_format_config_widget.figure_format_config
        else:
            # has non-GUI provider of figure_format_config
            figure_format_config = _curr_gui_figure_format_config_widget.figure_format_config

        if debug_print:
            print(f'recovered gui figure_format_config: {figure_format_config}')

        return figure_format_config

    figure_format_config = _get_curr_figure_format_config()

    ## PDF Output, NOTE this is single plot stuff: uses active_config_name
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_export_metadata

    filter_name = active_config_name
    _build_pdf_pages_output_info, programmatic_display_function_testing_output_parent_path = build_pdf_export_metadata(curr_active_pipeline.sess.get_description(), filter_name=filter_name)
    print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')
    
    
    ## Test getting figure save paths:
    _test_fig_path = curr_active_config.plotting_config.get_figure_save_path('test')
    print(f'_test_fig_path: {_test_fig_path}\n\t exists? {_test_fig_path.exists()}')

    return active_identifying_filtered_session_ctx, programmatic_display_function_testing_output_parent_path
    




class Plot(object):
    """a member dot accessor for display functions.

    Can call like: `plot._display_1d_placefields`

    """
    def __init__(self, curr_active_pipeline):
        super(Plot, self).__init__()
        self._pipeline_reference = curr_active_pipeline

    def __dir__(self):
        return self._pipeline_reference.registered_display_function_names # ['area', 'perimeter', 'location']
    
    def __getattr__(self, k):
        if '__getstate__' in k: # a trick to make spyder happy when inspecting dotdict
            def _dummy():
                pass
            return _dummy
        # return self[k]
        # return self._pipeline_reference.display(display_function=k, active_identifying_session_ctx=self._pipeline_reference.sess.get_context())
        return self._pipeline_reference.display(display_function=k, active_session_configuration_context=list(self._pipeline_reference.filtered_contexts.values())[-1])





# ==================================================================================================================== #
# 2022-08-18                                                                                                           #
# ==================================================================================================================== #

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
# Pre 2022-08-16 Figure Docking                                                                                        #
# ==================================================================================================================== #
from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.DockAreaWrapper import DockAreaWrapper
from pyphoplacecellanalysis.GUI.PyQtPlot.DockingWidgets.NestedDockAreaWidget import NestedDockAreaWidget

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
            _, dDisplayItem = active_containing_dockAreaWidget.add_display_dock(nested_out_widget_key, dockSize=(min_width, min_height), display_config=CustomDockDisplayConfig(showCloseButton=False), widget=all_nested_dock_area_widgets[filter_name], dockAddLocationOpts=dockAddLocationOpts)
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
                _, dDisplayItem = all_nested_dock_area_widgets[filter_name].add_display_dock(figure_key, dockSize=(fig_width, fig_height), display_config=CustomDockDisplayConfig(showCloseButton=False), widget=fig_window, dockAddLocationOpts=dockAddLocationOpts)
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
                    
                display_config = CustomDockDisplayConfig()
            
                _, dDisplayItem = active_containing_dockAreaWidget.add_display_dock(figure_key, dockSize=(fig_width, fig_height), display_config=CustomDockDisplayConfig(showCloseButton=False), widget=fig_window, dockAddLocationOpts=dockAddLocationOpts)
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
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.Specific3DTimeCurves import Specific3DTimeCurvesHelper
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
    from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.TimeCurves.Render3DTimeCurvesBaseGridMixin import BaseGrid3DTimeCurvesHelper, Render3DTimeCurvesBaseGridMixin


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

