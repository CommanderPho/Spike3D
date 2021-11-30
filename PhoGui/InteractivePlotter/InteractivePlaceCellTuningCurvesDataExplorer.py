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

from PhoGui.InteractivePlotter.shared_helpers import InteractivePyvistaPlotterBuildIfNeededMixin

# needs perform_plot_flat_arena
class InteractivePlaceCellTuningCurvesDataExplorer(InteractivePyvistaPlotterBuildIfNeededMixin):
    show_legend = True

    def __init__(self, active_config, x, y, active_epoch_placefields, pf_colors, extant_plotter=None):
        self.active_config = active_config
        self.x = x
        self.y = y
        self.active_epoch_placefields = active_epoch_placefields
        self.pf_colors = pf_colors
        
        # Initial setup
        # self.debug_console_widget = None
        self.pActiveTuningCurvesPlotter = extant_plotter
        
    
            
        
    def plot(self, pActiveTuningCurvesPlotter=None):
        ## Build the new BackgroundPlotter:
        self.pActiveTuningCurvesPlotter = InteractivePlaceCellTuningCurvesDataExplorer.build_new_plotter_if_needed(pActiveTuningCurvesPlotter)
        # Plot the flat arena
        perform_plot_flat_arena(self.pActiveTuningCurvesPlotter, self.x, self.y, bShowSequenceTraversalGradient=False)
        self.pActiveTuningCurvesPlotter, tuningCurvePlotActors, tuningCurvePlotLegendActor = plot_placefields2D(self.pActiveTuningCurvesPlotter, self.active_epoch_placefields, self.pf_colors, zScalingFactor=10.0, show_legend=True) 

        # Adds a multi-line debug console to the GUI for output logging:
        debug_console_widget = MultilineTextConsoleWidget(self.pActiveTuningCurvesPlotter)
        debug_console_widget.add_line_to_buffer('test log')
        # debug_console_widget.add_line_to_buffer('test log 2')
        # Adds a list of toggle checkboxe widgets to turn on and off each placemap
        use_mutually_exclusive_placefield_checkboxes = True
        if use_mutually_exclusive_placefield_checkboxes:
            checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks, mutually_exclusive_radiobutton_group = add_placemap_toggle_mutually_exclusive_checkboxes(self.pActiveTuningCurvesPlotter, tuningCurvePlotActors, self.pf_colors, active_element_idx=4, require_active_selection=False, is_debug=False)
        else:
            mutually_exclusive_radiobutton_group = None
            checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks = add_placemap_toggle_checkboxes(self.pActiveTuningCurvesPlotter, tuningCurvePlotActors, self.pf_colors, widget_check_states=False)
        return self.pActiveTuningCurvesPlotter
    
    
