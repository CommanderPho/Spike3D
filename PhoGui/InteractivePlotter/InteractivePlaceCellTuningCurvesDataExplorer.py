#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from PhoPositionalData.plotting.spikeAndPositions import plot_placefields2D, update_plotVisiblePlacefields2D, build_custom_placefield_maps_lookup_table
from PhoPositionalData.plotting.gui import SetVisibilityCallback, MutuallyExclusiveRadioButtonGroup, add_placemap_toggle_checkboxes, add_placemap_toggle_mutually_exclusive_checkboxes

from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper
from PhoGui.PhoCustomVtkWidgets import MultilineTextConsoleWidget

from PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data, perform_plot_flat_arena, build_spike_spawn_effect_light_actor, spike_geom_circle, spike_geom_box, spike_geom_cone, animal_location_circle, animal_location_trail_circle
#

from PhoGui.InteractivePlotter.shared_helpers import InteractiveDataExplorerBase

# needs perform_plot_flat_arena
class InteractivePlaceCellTuningCurvesDataExplorer(InteractiveDataExplorerBase): 
    """[summary]
    """
    show_legend = True

    def __init__(self, active_config, active_session, active_epoch_placefields, pf_colors, extant_plotter=None):
        super(InteractivePlaceCellTuningCurvesDataExplorer, self).__init__(active_config, active_session, extant_plotter, data_explorer_name='InteractivePlaceCellTuningCurvesDataExplorer')
        self.params.active_epoch_placefields = active_epoch_placefields
        self.params.pf_colors = pf_colors
        self._setup()

    
    def _setup_variables(self):
        num_cells, spike_list, cell_ids, flattened_spike_identities, flattened_spike_times, flattened_sort_indicies, t_start, reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds, self.params.flattened_spike_positions_list = InteractiveDataExplorerBase._unpack_variables(self.active_session)
        ### Build the flattened spike positions list
        # Determine the x and y positions each spike occured for each cell
        ## new_df style:
        self.debug.flattened_spike_positions_list_new = self.active_session.flattened_spiketrains.spikes_df[["x", "y"]].to_numpy().T

        ## old-style:
        self.debug.spike_positions_list_old = self.params.flattened_spike_positions_list


    def _setup_visualization(self):        
        pass
    
    


            
        
    def plot(self, pActivePlotter=None):
        ## Build the new BackgroundPlotter:
        self.p = InteractivePlaceCellTuningCurvesDataExplorer.build_new_plotter_if_needed(pActivePlotter)
        # Plot the flat arena
        self.plots['maze_bg'] = perform_plot_flat_arena(self.p, self.x, self.y, bShowSequenceTraversalGradient=False)
        
        self.p, tuningCurvePlotActors, tuningCurvePlotLegendActor = plot_placefields2D(self.p, self.params.active_epoch_placefields, self.params.pf_colors, zScalingFactor=10.0, show_legend=True) 

        # Adds a multi-line debug console to the GUI for output logging:
        debug_console_widget = MultilineTextConsoleWidget(self.p)
        debug_console_widget.add_line_to_buffer('test log')
        # debug_console_widget.add_line_to_buffer('test log 2')
        # Adds a list of toggle checkboxe widgets to turn on and off each placemap
        use_mutually_exclusive_placefield_checkboxes = True
        if use_mutually_exclusive_placefield_checkboxes:
            checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks, mutually_exclusive_radiobutton_group = add_placemap_toggle_mutually_exclusive_checkboxes(self.p, tuningCurvePlotActors, self.params.pf_colors, active_element_idx=4, require_active_selection=False, is_debug=False)
        else:
            mutually_exclusive_radiobutton_group = None
            checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks = add_placemap_toggle_checkboxes(self.p, tuningCurvePlotActors, self.params.pf_colors, widget_check_states=False)
        return self.p
    
    
    # def rough_add_spikes(self, sesssion):
        
    #     active_included_recent_only_indicies = ((flattened_spike_times > recent_spikes_t_start) & (flattened_spike_times < t_stop)) # Two Sided Range Mode
    #     # active_included_recent_only_indicies = ((flattened_spikes.flattened_spike_times > t_start) & (flattened_spikes.flattened_spike_times < t_stop)) # Two Sided Range Mode
    #     recent_only_spikes_pdata, recent_only_spikes_pc = build_active_spikes_plot_data(flattened_spike_times[active_included_recent_only_indicies],
    #                                                                                     flattened_spike_active_unitIdentities[active_included_recent_only_indicies],
    #                                                                                     flattened_spike_positions_list[:, active_included_recent_only_indicies],
    #                                                                                     spike_geom=spike_geom_cone.copy())
    #     if recent_only_spikes_pc.n_points >= 1:
    #         self.plots['spikes_main_recent_only'] = self.p.add_mesh(recent_only_spikes_pc, name='recent_only_spikes_main', scalars='cellID', cmap=self.active_config.plotting_config.active_cells_listed_colormap, show_scalar_bar=False, lighting=False, render=False) # color='white'