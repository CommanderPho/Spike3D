### Includes
"""
@author: pho
"""
import sys
import importlib
from pathlib import Path
import logging

# from pyphocorehelpers.print_helpers import build_module_logger
# module_logger = build_module_logger('Spike3D.notebook')

from copy import deepcopy

from numba import jit
import numpy as np
import pandas as pd

# # required to enable non-blocking interaction:
# %gui qt5

import pyphoplacecellanalysis.External.pyqtgraph as pg # Used to plot Speed vs. Pf Desnity Metrics
from pyphoplacecellanalysis.External.pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyphoplacecellanalysis.External.pyqtgraph.opengl as gl # for 3D raster plot

# Pho's Formatting Preferences
from pyphocorehelpers.preferences_helpers import set_pho_preferences, set_pho_preferences_concise, set_pho_preferences_verbose
set_pho_preferences_concise()

## Pho's Custom Libraries:
from pyphocorehelpers.general_helpers import get_arguments_as_optional_dict, inspect_callable_arguments
from pyphocorehelpers.indexing_helpers import partition, compute_spanning_bins, interleave_elements, compute_position_grid_size, compute_paginated_grid_config
from pyphocorehelpers.print_helpers import PrettyPrintable, WrappingMessagePrinter, print_value_overview_only, print_keys_if_possible, debug_dump_object_member_shapes, safe_get_variable_shape
from pyphocorehelpers.DataStructure.dynamic_parameters import DynamicParameters
from pyphocorehelpers.performance_timing_helpers import WrappingPerformanceTimer
from pyphocorehelpers.gui.interaction_helpers import CallbackWrapper
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphocorehelpers.indexing_helpers import BinningInfo, compute_spanning_bins, get_bin_centers, get_bin_edges, debug_print_1D_bin_infos, interleave_elements, build_spanning_grid_matrix

# pyPhoPlaceCellAnalysis:
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import NeuropyPipeline # get_neuron_identities
# from pyphoplacecellanalysis.General.SessionSelectionAndFiltering import batch_filter_session, build_custom_epochs_filters
from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties

from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.ComputationFunctionRegistryHolder import ComputationFunctionRegistryHolder
from pyphoplacecellanalysis.General.Pipeline.Stages.DisplayFunctions.DisplayFunctionRegistryHolder import DisplayFunctionRegistryHolder
from pyphocorehelpers.gui.PhoUIContainer import PhoUIContainer
from pyphoplacecellanalysis.GUI.Qt.Menus.PhoMenuHelper import PhoMenuHelper
from pyphoplacecellanalysis.GUI.Qt.Menus.LocalMenus_AddRenderable import LocalMenus_AddRenderable
from pyphoplacecellanalysis.GUI.Qt.Menus.SpecificMenus.CreateLinkedWidget_MenuProvider import CreateLinkedWidget_MenuProvider

# NeuroPy (Diba Lab Python Repo) Loading
# from neuropy import core
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.core.session.dataSession import DataSession
from neuropy.analyses.placefields import PlacefieldComputationParameters

from neuropy.core.laps import Laps  # Used for adding laps in KDiba mode
from neuropy.utils.efficient_interval_search import get_non_overlapping_epochs, drop_overlapping # Used for adding laps in KDiba mode

from neuropy.core.epoch import NamedTimerange

from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder, DataSessionFormatBaseRegisteredClass
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
from neuropy.core.session.Formats.Specific.HiroDataSessionFormat import HiroDataSessionFormatRegisteredClass

## Plotting Helpers:
from pyphoplacecellanalysis.GUI.PyQtPlot.pyqtplot_Matrix import MatrixRenderingWindow
from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow
from pyphoplacecellanalysis.Pho2D.PyQtPlots.plot_placefields import pyqtplot_plot_image_array
from pyphoplacecellanalysis.Pho2D.PyQtPlots.Extensions.pyqtgraph_helpers import pyqtplot_build_image_bounds_extent, pyqtplot_plot_image

known_data_session_type_properties_dict = DataSessionFormatRegistryHolder.get_registry_known_data_session_type_dict()
active_data_session_types_registered_classes_dict = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()
# DataSessionFormatRegistryHolder.get_registry()

enable_saving_to_disk = False
# common_parent_foldername = Path(r'R:\Dropbox (Personal)\Active\Kamran Diba Lib\Pho-Kamran-Meetings\Final Placemaps 2021-01-14')
common_parent_foldername = Path(r'R:\Dropbox (Personal)\Active\Kamran Diba Lib\Pho-Kamran-Meetings\2022-01-16')

# np.set_string_function(lambda x:f'np.array[{np.shape(x)}]', repr=True)
# np.set_string_function(None, repr=True)
from pyphocorehelpers.general_helpers import safely_accepts_kwargs
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import loadData, saveData

## For computation parameters:
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.utils.dynamic_container import DynamicContainer,
from neuropy.utils.mixins.indexing_helpers import get_dict_subset, override_dict, overriding_dict_with, overriding_dict_with, get_dict_subset
from neuropy.utils.position_util import compute_position_grid_size

## Test Matplotlib plotting:
from neuropy.utils.matplotlib_helpers import _build_variable_max_value_label, add_inner_title, enumTuningMap2DPlotMode, _build_square_checkerboard_image, enumTuningMap2DPlotVariables, compute_data_aspect_ratio

def build_eloy_computation_configs(sess, **kwargs):
    """ OPTIONALLY can be overriden by implementors to provide specific filter functions """
    # (4.0, 4.0)cm bins, (6.0, 6.0)cm gaussian smoothing
    # peak frate > 2Hz 
    # return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(4.0, 4.0), smooth=(6.0, 6.0), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None), spike_analysis=None)]
    # return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(4.0, 4.0), smooth=(2.5, 2.5), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None), spike_analysis=None)]
    # return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(4.0, 4.0), smooth=(0.2, 0.2), frate_thresh=0.2, time_bin_size=1.0, computation_epochs = None), spike_analysis=None)]
    return [DynamicContainer(pf_params=PlacefieldComputationParameters(speed_thresh=10.0, grid_bin=(4.0, 4.0), smooth=(0.2, 0.2), frate_thresh=0.2, time_bin_size=0.025, computation_epochs = None), spike_analysis=None)]



### Bapun Open-Field Experiment (2022-08-09 Analysis)
from neuropy.core.session.SessionSelectionAndFiltering import build_custom_epochs_filters # used particularly to build Bapun-style filters

active_data_mode_name = 'bapun'
active_data_mode_registered_class = active_data_session_types_registered_classes_dict[active_data_mode_name]
active_data_mode_type_properties = known_data_session_type_properties_dict[active_data_mode_name]
basedir = r'W:\Data\Bapun\RatN\Day4OpenField'

curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties, override_basepath=Path(basedir), force_reload=False)
# active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess, epoch_name_includelist=['maze','sprinkle']) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
# active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['maze','sprinkle'])
active_session_filter_configurations = build_custom_epochs_filters(curr_active_pipeline.sess, epoch_name_includelist=['maze','sprinkle'])
# active_session_filter_configurations = active_data_mode_registered_class.build_filters_pyramidal_epochs(curr_active_pipeline.sess, epoch_name_includelist=['maze','sprinkle'])
active_session_computation_configs = active_data_mode_registered_class.build_default_computation_configs(sess=curr_active_pipeline.sess, time_bin_size=1.0)
curr_active_pipeline.filter_sessions(active_session_filter_configurations)
curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_excludelist=['_perform_spike_burst_detection_computation', '_perform_velocity_vs_pf_density_computation', '_perform_velocity_vs_pf_simplified_count_density_computation']) # SpikeAnalysisComputations._perform_spike_burst_detection_computation
curr_active_pipeline.prepare_for_display(root_output_dir=r'W:\Data\Output', should_smooth_maze=True) # TODO: pass a display config
# curr_active_pipeline.save_pipeline()












### Plot Setup:

active_identifying_session_ctx = curr_active_pipeline.sess.get_context() # 'bapun_RatN_Day4_2019-10-15_11-30-06'

## MATPLOTLIB Imports:
import matplotlib
# configure backend here
matplotlib.use('Qt5Agg')
# backend_qt5agg
# matplotlib.use('AGG') # non-interactive backend
## 2022-08-16 - Surprisingly this works to make the matplotlib figures render only to .png file, not appear on the screen!
import matplotlib.pyplot as plt
import matplotlib as mpl
from neuropy.utils.matplotlib_helpers import enumTuningMap2DPlotVariables # for getting the variant name from the dict
_bak_rcParams = mpl.rcParams.copy()
mpl.rcParams['toolbar'] = 'None' # disable toolbars

debug_print = False
# print(f'curr_active_pipeline.filtered_contexts: {curr_active_pipeline.filtered_contexts}')
# print(f'curr_active_pipeline.display_output: {curr_active_pipeline.display_output}')

# ## Old way:
# # Create `master_dock_win` - centralized plot output window to collect individual figures/controls in (2022-08-18) 
# out_items = {}
# master_dock_win, app, out_items = context_nested_docks(curr_active_pipeline, enable_gui=False, debug_print=True)


### 🟢✅ SETTING THE ACTIVE CONFIG NAME:

# active_config_name = 'maze1'
# active_config_name = 'maze2'
# active_config_name = 'maze'
# active_config_name = 'sprinkle'

# active_config_name = 'maze_PYR'

# active_config_name = 'maze1_rippleOnly'
# active_config_name = 'maze2_rippleOnly'

active_config_name = curr_active_pipeline.active_config_names[0] # get the first name by default
# active_config_name = curr_active_pipeline.active_config_names[-1] # get the last name
print(f'active_config_name: {active_config_name}')

## Add the filter to the active context
# active_identifying_filtered_session_ctx = active_identifying_session_ctx.adding_context('filter', filter_name=active_config_name) # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'
active_identifying_filtered_session_ctx = curr_active_pipeline.filtered_contexts[active_config_name] # 'bapun_RatN_Day4_2019-10-15_11-30-06_maze'

# Get relevant variables:
# curr_active_pipeline is set above, and usable here
sess: DataSession = curr_active_pipeline.filtered_sessions[active_config_name]

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
active_one_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_Decoder', None)
active_two_step_decoder = curr_active_pipeline.computation_results[active_config_name].computed_data.get('pf2D_TwoStepDecoder', None)
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
display_output = active_display_output | curr_active_pipeline.display('_display_context_nested_docks', active_identifying_session_ctx, enable_gui=True, debug_print=True) # returns {'master_dock_win': master_dock_win, 'app': app, 'out_items': out_items}
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
from matplotlib.backends import backend_pdf, backend_pgf, backend_ps
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_pdf_export_metadata

filter_name = active_config_name
_build_pdf_pages_output_info, programmatic_display_function_testing_output_parent_path = build_pdf_export_metadata(curr_active_pipeline.sess.get_description(), filter_name=filter_name)
print(f'Figure Output path: {str(programmatic_display_function_testing_output_parent_path)}')


# decoding_time_bin_size = new_2D_decoder.time_bin_size
# decoding_time_bin_size = 0.01
# decoding_time_bin_size = 0.1
decoding_time_bin_size = 1.0
# decoding_time_bin_size = 0.2

## Testing PBE Decoding
active_decoder = active_one_step_decoder
# active_decoder = new_2D_decoder
# active_decoder = new_1D_decoder

# active_epoch = curr_active_config.active_epochs.to_Epoch() # <NamedTimerange: {'name': 'maze1', 'start_end_times': array([22.26, 1211.56])};>
active_epoch = sess.ripple.to_dataframe()
active_epoch_decoder_result = active_decoder.decode_specific_epochs(sess.spikes_df, filter_epochs=active_epoch, decoding_time_bin_size=decoding_time_bin_size, debug_print=False)
# active_epoch_decoder_result

active_time_bins = active_epoch_decoder_result.time_bin_edges[0]
active_posterior_container = active_epoch_decoder_result.marginal_x_list[0]
active_posterior = active_posterior_container.p_x_given_n
active_most_likely_positions = active_posterior_container.most_likely_positions_1D

