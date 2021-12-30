## This file serves as overflow from active Jupyter-lab notebooks, to eventually be refactored.
from copy import deepcopy
from matplotlib.colors import ListedColormap
from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt # conda install -c conda-forge pyvistaqt

from PhoPositionalData.analysis.interactive_placeCell_config import InteractivePlaceCellConfig, VideoOutputModeConfig, PlottingConfig  # VideoOutputModeConfig, InteractivePlaceCellConfigs
from PhoPositionalData.analysis.interactive_placeCell_config import print_subsession_neuron_differences


from neuropy.core import Laps

# from neuropy.analyses import perform_compute_placefields
from neuropy.analyses.placefields import PlacefieldComputationParameters
from neuropy.analyses.pho_custom_placefields import PfND

from neuropy.analyses.laps import estimate_laps, compute_laps_spike_indicies

from neuropy.utils.colors_util import get_neuron_colors
from neuropy.plotting.placemaps import plot_all_placefields

from PhoPositionalData.plotting.laps import plot_laps_2d

should_force_recompute_placefields = True
should_display_2D_plots = True




def compute_placefields_masked_by_epochs(sess, active_config, included_epochs=None, should_display_2D_plots=False):
    active_session = deepcopy(sess)
    active_epoch_placefields1D, active_epoch_placefields2D = compute_placefields_as_needed(active_session, active_config.computation_config, active_config, None, None, included_epochs=included_epochs, should_force_recompute_placefields=True, should_display_2D_plots=should_display_2D_plots)
    # Focus on the 2D placefields:
    # active_epoch_placefields = active_epoch_placefields2D
    # Get the updated session using the units that have good placefields
    # active_session, active_config, good_placefield_neuronIDs = process_by_good_placefields(active_session, active_config, active_epoch_placefields)
    # debug_print_spike_counts(active_session)
    return active_epoch_placefields1D, active_epoch_placefields2D


    

def perform_compute_placefields(active_session_spikes_df, active_pos, computation_config: PlacefieldComputationParameters, active_epoch_placefields1D=None, active_epoch_placefields2D=None, included_epochs=None, should_force_recompute_placefields=True):
    """ Computes both 1D and 2D placefields.
    active_epoch_session_Neurons: 
    active_epoch_pos: a Position object
    included_epochs: a Epoch object to filter with, only included epochs are included in the PF calculations
    active_epoch_placefields1D (Pf1D, optional) & active_epoch_placefields2D (Pf2D, optional): allow you to pass already computed Pf1D and Pf2D objects from previous runs and it won't recompute them so long as should_force_recompute_placefields=False, which is useful in interactive Notebooks/scripts
    Usage:
        active_epoch_placefields1D, active_epoch_placefields2D = perform_compute_placefields(active_epoch_session_Neurons, active_epoch_pos, active_epoch_placefields1D, active_epoch_placefields2D, active_config.computation_config, should_force_recompute_placefields=True)
    """
    ## Linearized (1D) Position Placefields:
    if ((active_epoch_placefields1D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_placefields...', end=' ')
        # active_epoch_placefields1D = Pf1D(neurons=active_session_Neurons, position=deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
        #                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
        #                                   grid_bin=computation_config.grid_bin_1D, smooth=computation_config.smooth_1D)
        # PfND version:
        active_epoch_placefields1D = PfND(deepcopy(active_session_spikes_df), deepcopy(active_pos.linear_pos_obj), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields1D already exists, reusing it.')

    ## 2D Position Placemaps:
    if ((active_epoch_placefields2D is None) or should_force_recompute_placefields):
        print('Recomputing active_epoch_placefields2D...', end=' ')
        # active_epoch_placefields2D = Pf2D(neurons=active_session_Neurons, position=deepcopy(active_pos), epochs=included_epochs,
        #                                   speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
        #                                   grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)
        # PfND version:
        active_epoch_placefields2D = PfND(deepcopy(active_session_spikes_df), deepcopy(active_pos), epochs=included_epochs,
                                          speed_thresh=computation_config.speed_thresh, frate_thresh=computation_config.frate_thresh,
                                          grid_bin=computation_config.grid_bin, smooth=computation_config.smooth)

        print('\t done.')
    else:
        print('active_epoch_placefields2D already exists, reusing it.')
    
    return active_epoch_placefields1D, active_epoch_placefields2D




def compute_placefields_as_needed(active_session, computation_config=None, general_config: InteractivePlaceCellConfig=None, active_placefields1D = None, active_placefields2D = None, included_epochs=None, should_force_recompute_placefields=False, should_display_2D_plots=False):
    if computation_config is None:
        computation_config = PlacefieldComputationParameters(speed_thresh=9, grid_bin=2, smooth=0.5)
    # active_placefields1D, active_placefields2D = perform_compute_placefields(active_session.neurons, active_session.position, computation_config, active_placefields1D, active_placefields2D, included_epochs=included_epochs, should_force_recompute_placefields=True)
    active_placefields1D, active_placefields2D = perform_compute_placefields(active_session.spikes_df, active_session.position, computation_config, active_placefields1D, active_placefields2D, included_epochs=included_epochs, should_force_recompute_placefields=True)
    # Plot the placefields computed and save them out to files:
    if should_display_2D_plots:
        ax_pf_1D, occupancy_fig, active_pf_2D_figures, active_pf_2D_gs = plot_all_placefields(active_placefields1D, active_placefields2D, general_config)
    else:
        print('skipping 2D placefield plots')
    return active_placefields1D, active_placefields2D




## Plotting Colors:
def build_units_colormap(neuron_ids):
    """ 
    Usage:
        pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    """
    pf_sort_ind = np.array([int(i) for i in np.arange(len(neuron_ids))]) # convert to integer scalar array
    pf_colors = get_neuron_colors(pf_sort_ind, cmap=None) # [4 x n_neurons]: colors are by ascending index ID
    pf_colormap = pf_colors.T # [n_neurons x 4] Make the colormap from the listed colors, used seemingly only by 'runAnalysis_PCAandICA(...)'
    pf_listed_colormap = ListedColormap(pf_colormap)
    return pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap



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

## For building the configs used to filter the session by epoch: 
def build_configs(session_config, active_epoch, active_subplots_shape = (1,1)):
    ## Get the config corresponding to this epoch/session settings:
    active_config = InteractivePlaceCellConfig(active_session_config=session_config, active_epochs=active_epoch, video_output_config=None, plotting_config=None) # '3|1    

    active_config.video_output_config = VideoOutputModeConfig(active_frame_range=np.arange(100.0, 120.0), 
                                                              video_output_parent_dir=Path('output', session_config.session_name, active_epoch.name),
                                                              active_is_video_output_mode=False)
    active_config.plotting_config = PlottingConfig(output_subplots_shape=active_subplots_shape,
                                                   output_parent_dir=Path('output', session_config.session_name, active_epoch.name)
                                                  )
    # Make the directories:
    active_config.plotting_config.active_output_parent_dir.mkdir(parents=True, exist_ok=True) # makes the directory if it isn't already there
    return active_config


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



def estimation_session_laps(sess, N=20):
    """ 2021-12-21 - Pho's lap estimation from the position data (only)
    Replaces the sess.laps which is computed or loaded from the spikesII.mat spikes data (which isn't very good)"""
    # backup the extant laps object to prepare for the new one:
    sess.old_laps_obj = deepcopy(sess.laps)
    # plot originals:
    fig, out_axes_list = plot_laps_2d(sess, legacy_plotting_mode=True)
    out_axes_list[0].set_title('Old SpikeII computed Laps')
    position_obj = sess.position
    # position_obj.dt
    position_obj.compute_higher_order_derivatives()
    pos_df = position_obj.compute_smoothed_position_info(N=N) ## Smooth the velocity curve to apply meaningful logic to it
    pos_df = position_obj.to_dataframe()
    # custom_test_laps = deepcopy(sess.laps)
    spikes_df = deepcopy(sess.spikes_df)

    desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings = estimate_laps(pos_df)
    custom_test_laps_obj = Laps.from_estimated_laps(pos_df['t'].to_numpy(), desc_crossing_beginings, desc_crossing_endings, asc_crossing_beginings, asc_crossing_endings)
    ## Determine the spikes included with each computed lap:
    custom_test_laps_obj = compute_laps_spike_indicies(custom_test_laps_obj, spikes_df)
    sess.laps = deepcopy(custom_test_laps_obj) # replace the laps obj
    # plot computed:
    fig, out_axes_list = plot_laps_2d(sess, legacy_plotting_mode=False)
    out_axes_list[0].set_title('New Pho Position Thresholding Estimated Laps')

    return sess

