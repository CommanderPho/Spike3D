#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
from copy import deepcopy
import numpy as np
import pandas as pd
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from matplotlib.colors import ListedColormap, to_hex

from scipy.interpolate import RectBivariateSpline # for 2D spline interpolation

from PhoGui.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter
# from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.general_plotting_mixins
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.occupancy_plotting_mixins import OccupancyPlottingMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.placefield_plotting_mixins import HideShowPlacefieldsRenderingMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.spikes_mixins import SpikesDataframeOwningMixin, SpikeRenderingMixin, HideShowSpikeRenderingMixin

from pyphoplacecellanalysis.PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data_df, plot_placefields2D, update_plotVisiblePlacefields2D, build_custom_placefield_maps_lookup_table
from pyphoplacecellanalysis.PhoPositionalData.plotting.gui import CallbackSequence, SetVisibilityCallback, MutuallyExclusiveRadioButtonGroup, add_placemap_toggle_checkboxes, add_placemap_toggle_mutually_exclusive_checkboxes

from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper
from PhoGui.PhoCustomVtkWidgets import MultilineTextConsoleWidget

from pyphoplacecellanalysis.PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data, perform_plot_flat_arena, build_spike_spawn_effect_light_actor, spike_geom_circle, spike_geom_box, spike_geom_cone, animal_location_circle, animal_location_trail_circle
#
from PhoGui.InteractivePlotter.shared_helpers import InteractiveDataExplorerBase


# needs perform_plot_flat_arena
class InteractivePlaceCellTuningCurvesDataExplorer(OccupancyPlottingMixin, HideShowPlacefieldsRenderingMixin, SpikesDataframeOwningMixin, SpikeRenderingMixin, HideShowSpikeRenderingMixin, InteractiveDataExplorerBase): 
    """[summary]
    """
    show_legend = True

    def __init__(self, active_config, active_session, active_epoch_placefields, pf_colors, extant_plotter=None):
        super(InteractivePlaceCellTuningCurvesDataExplorer, self).__init__(active_config, active_session, extant_plotter, data_explorer_name='TuningMapDataExplorer')
        self.params.active_epoch_placefields = deepcopy(active_epoch_placefields)
        self.params.pf_colors = deepcopy(pf_colors)
        self.params.pf_colors_hex = None
        self.params.pf_active_configs = None
        self.gui = dict()
        
        self.use_unit_id_as_cell_id = False # if False, uses the normal 'aclu' value as the cell id (which I think is correct)
        
        self._setup()

    # from NeuronIdentityAccessingMixin
    @property
    def neuron_ids(self):
        """ an alias for self.cell_ids required for NeuronIdentityAccessingMixin """
        return self.cell_ids 
    
    @property
    def cell_ids(self):
        """ e.g. the list of valid cell_ids (unique aclu values) """
        return np.array(self.params.cell_ids) 
    
    
    def _setup_variables(self):
        num_cells, spike_list, self.params.cell_ids, self.params.flattened_spike_identities, self.params.flattened_spike_times, flattened_sort_indicies, t_start, self.params.reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds, self.params.flattened_spike_positions_list = InteractiveDataExplorerBase._unpack_variables(self.active_session)
        ## Ensure we have the 'unit_id' property
        if self.use_unit_id_as_cell_id:
            try:
                test = self.active_session.spikes_df['unit_id']
            except KeyError as e:
                # build the valid key:
                self.active_session.spikes_df['unit_id'] = np.array([int(self.active_session.neurons.reverse_cellID_index_map[original_cellID]) for original_cellID in self.active_session.spikes_df['aclu'].values])
        else:
            assert ('aclu' in self.active_session.spikes_df.columns), "self.active_session.spikes_df must contain the 'aclu' column! Something is wrong!"     


    def _setup_visualization(self): 
        self.params.debug_disable_all_gui_controls = True
        
        self.params.enable_placefield_aligned_spikes = True # If True, the spikes are aligned to the z-position of their respective place field, so they visually sit on top of the placefield surface
        # self.params.zScalingFactor = 10.0
        self.params.zScalingFactor = 100.0
        
        self.params.use_mutually_exclusive_placefield_checkboxes = True       
        self.params.show_legend = True
        
        # self.params.use_unit_id_slider_instead_of_checkboxes = True
        self.params.use_unit_id_slider_instead_of_checkboxes = False
    
        self.params.use_dynamic_spike_opacity_for_hiding = True
        
        if self.params.use_dynamic_spike_opacity_for_hiding:
            self.setup_hide_show_spike_rendering_mixin()
    
        if self.params.enable_placefield_aligned_spikes:
            # compute the spike z-positions from the placefield2D objects if that option is selected.
            # self._compute_z_position_spike_offsets()
            # self._compute_z_position_spike_offsets()
            pass

        self.params.pf_colors_hex = [to_hex(self.params.pf_colors[:,i], keep_alpha=False) for i in self.tuning_curve_indicies]
        self.setup_spike_rendering_mixin()
        self.build_tuning_curve_configs()
        self.setup_occupancy_plotting_mixin()
    
    @property
    def pf_names(self):
        return self.params.cell_ids
        # return self.active_session.neurons.neuron_ids
            
        
    def plot(self, pActivePlotter=None):
        ## Build the new BackgroundPlotter:
        self.p = InteractivePlaceCellTuningCurvesDataExplorer.build_new_plotter_if_needed(pActivePlotter, title=self.data_explorer_name)
        # Plot the flat arena
        self.plots['maze_bg'] = perform_plot_flat_arena(self.p, self.x, self.y, bShowSequenceTraversalGradient=False)
        
        self.p, self.plots['tuningCurvePlotActors'], self.plots_data['tuningCurvePlotData'], self.plots['tuningCurvePlotLegendActor'], temp_plots_data = plot_placefields2D(self.p, self.params.active_epoch_placefields, self.params.pf_colors, zScalingFactor=self.params.zScalingFactor, show_legend=self.params.show_legend) 
        # Build the widget labels:
        self.params.unit_labels = temp_plots_data['unit_labels'] # fetch the unit labels from the extra data dict.
        self.params.pf_unit_ids = temp_plots_data['good_placefield_neuronIDs'] # fetch the unit labels from the extra data dict.
        ## TODO: For these, we actually want the placefield value as the Z-positions, will need to unwrap them or something (maybe .ravel(...)?)
        ## TODO: also need to add in the checkbox functionality to hide/show only the spikes for the highlighted units
        # .threshold().elevation()
        
        # hide the tuning curves automatically on startup (they don't render correctly anyway):
        self._hide_all_tuning_curves()
        
        # active_spike_index = 4
        # active_included_place_cell_spikes_indicies = self.active_session.spikes_df.eval('(unit_id == @active_spike_index)') # '@' prefix indicates a local variable. All other variables are evaluated as column names
        needs_render = self.plot_spikes()

        if needs_render:
            self.p.render()

        # Adds a multi-line debug console to the GUI for output logging:        
        self.gui['debug_console_widget'] = MultilineTextConsoleWidget(self.p)
        self.gui['debug_console_widget'].add_line_to_buffer('test log')
        # debug_console_widget.add_line_to_buffer('test log 2')
        # Adds a list of toggle checkboxe widgets to turn on and off each placemap
        # self.setup_visibility_checkboxes(self.plots['tuningCurvePlotActors'])
        
        if not self.params.debug_disable_all_gui_controls:
            # build the visibility callbacks that will be used to update the meshes from the UI elements:
            self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'] = self.__build_callbacks(self.plots['tuningCurvePlotActors'])
            
            if self.params.use_unit_id_slider_instead_of_checkboxes:
                # use the discrete slider widget instead of the checkboxes
                self.__setup_visibility_slider_widget()
            else:
                # checkbox mode for unit ID selection: 
                self.__setup_visibility_checkboxes()
        else:
            print('self.params.debug_disable_all_gui_controls is True, so no gui controls will be built.')
        
        return self.p
    
    
    
    
    def __build_callbacks(self, tuningCurvePlotActors):
        combined_active_pf_update_callbacks = []
        for i, an_actor in enumerate(tuningCurvePlotActors):
            # Make a separate callback for each widget
            curr_visibility_callback = SetVisibilityCallback(an_actor)
            curr_spikes_update_callback = (lambda is_visible, i_copy=i: self._update_placefield_spike_visibility([i_copy], is_visible))
            combined_active_pf_update_callbacks.append(CallbackSequence([curr_visibility_callback, curr_spikes_update_callback]))
        return combined_active_pf_update_callbacks
            
            
    
    def __setup_visibility_checkboxes(self):
        # self.gui['tuningCurveSpikeVisibilityCallbacks'] = [lambda i: self.hide_placefield_spikes(i) for i in np.arange(len(tuningCurvePlotActors))]
        # self.gui['tuningCurveSpikeVisibilityCallbacks'] = [lambda is_visible: self.update_placefield_spike_visibility([i], is_visible) for i in np.arange(len(tuningCurvePlotActors))]
        # self.gui['tuningCurveSpikeVisibilityCallbacks'] = [lambda is_visible, i_copy=i: self._update_placefield_spike_visibility([i_copy], is_visible) for i in np.arange(len(tuningCurvePlotActors))]
        
        if self.params.use_mutually_exclusive_placefield_checkboxes:
            self.gui['checkboxWidgetActors'], self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'], self.gui['mutually_exclusive_radiobutton_group'] = add_placemap_toggle_mutually_exclusive_checkboxes(self.p, self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'], self.params.pf_colors, active_element_idx=4, require_active_selection=False, is_debug=False, additional_callback_actions=None, labels=self.params.unit_labels)
        else:
            self.gui['mutually_exclusive_radiobutton_group'] = None           
            self.gui['checkboxWidgetActors'], self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'] = add_placemap_toggle_checkboxes(self.p, self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'], self.params.pf_colors, widget_check_states=False, additional_callback_actions=None, labels=self.params.unit_labels)
        

       
    def __setup_visibility_slider_widget(self):
        # safe_integer_wrapper = lambda integer_local_idx: self._update_placefield_spike_visibility([int(integer_local_idx)])
        safe_integer_wrapper = lambda integer_local_idx: self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks']([int(integer_local_idx)])
        self.gui['interactive_unitID_slider_actor'] = PhoWidgetHelper.add_discrete_slider_widget(self.p, safe_integer_wrapper, [0, (len(self.gui['tuningCurveCombinedAllPlotActorsVisibilityCallbacks'])-1)], value=0, title='Selected Unit',event_type='end')
        ## I don't think this does anything:
        interactive_plotter = PhoInteractivePlotter(pyvista_plotter=self.p, interactive_timestamp_slider_actor=self.gui['interactive_unitID_slider_actor'])
        
        
        
    def _compute_z_position_spike_offsets(self):
        ## UNUSED?
        ## Potentially successfully implemented the z-interpolation!!!: 2D interpolation where the (x,y) point of each spike is evaluated to determine the Z-position it would correspond to on the pf map.
        # _spike_pf_heights_2D_splineAproximator = [RectBivariateSpline(active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers, active_epoch_placefields2D.ratemap.normalized_tuning_curves[i]) for i in np.arange(active_epoch_placefields2D.ratemap.n_neurons)] 
        
        _spike_pf_heights_2D_splineAproximator = [RectBivariateSpline(self.params.active_epoch_placefields.ratemap.xbin_centers, self.params.active_epoch_placefields.ratemap.ybin_centers, self.params.active_epoch_placefields.ratemap.tuning_curves[i]) for i in np.arange(self.params.active_epoch_placefields.ratemap.n_neurons)] 
        # active_epoch_placefields2D.spk_pos[i][0] and active_epoch_placefields2D.spk_pos[i][1] seem to successfully get the x and y data for the spike_pos[i]
        spike_pf_heights_2D = [_spike_pf_heights_2D_splineAproximator[i](self.params.active_epoch_placefields.spk_pos[i][0], self.params.active_epoch_placefields.spk_pos[i][1], grid=False) for i in np.arange(self.params.active_epoch_placefields.ratemap.n_neurons)] # the appropriately interpolated values for where the spikes should be on the tuning_curve

        # Attempt to set the spike heights:
        # Add a custom z override for the spikes but with the default value so nothing is changed:
        self.active_session.spikes_df['z'] = np.full_like(self.active_session.spikes_df['x'].values, 1.1) # Offset a little bit in the z-direction so we can see it

        for i in np.arange(self.params.active_epoch_placefields.ratemap.n_neurons):
            curr_cell_id = self.params.active_epoch_placefields.cell_ids[i]
            # set the z values for the current cell index to the heights offset for that cell:
            self.active_session.spikes_df.loc[(self.active_session.spikes_df.aclu == curr_cell_id), 'z'] = spike_pf_heights_2D[i] # Set the spike heights to the appropriate z value

        # when finished, self.active_session.spikes_df is modified with the updated 'z' values


