#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
import numpy as np
import pyvista as pv
from pyvistaqt.plotting import MultiPlotter

from pyphoplacecellanalysis.PhoPositionalData.plotting.animations import make_mp4_from_plotter

from PhoGui.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter

from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper
from PhoGui.PhoCustomVtkWidgets import MultilineTextConsoleWidget

from pyphoplacecellanalysis.PhoPositionalData.plotting.gui import customize_default_pyvista_theme, print_controls_helper_text
from pyphoplacecellanalysis.PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data, perform_plot_flat_arena, build_spike_spawn_effect_light_actor, spike_geom_circle, spike_geom_box, spike_geom_cone, animal_location_circle, animal_location_trail_circle
from PhoGui.InteractivePlotter.shared_helpers import InteractiveDataExplorerBase, InteractivePyvistaPlotterBuildIfNeededMixin, InteractivePyvistaPlotter_ObjectManipulationMixin
from pyphoplacecellanalysis.PhoPositionalData.plotting.visualization_window import VisualizationWindow # Used to build "Windows" into the data points such as the window defining the fixed time period preceeding the current time where spikes had recently fired, etc.
from numpy.lib.stride_tricks import sliding_window_view


class InteractiveCustomDataExplorer(InteractiveDataExplorerBase):
    """[summary]
    """
    def __init__(self, active_config, active_session, extant_plotter=None):
        # super().__init__(active_config, active_session, extant_plotter)
        super(InteractiveCustomDataExplorer, self).__init__(active_config, active_session, extant_plotter, data_explorer_name='CustomDataExplorer')
        self._setup()

    
    def _setup_variables(self):
        pass

    def _setup_visualization(self):
        pass


    ######################
    # General Plotting Method:
   
    # pf_colors, active_config
    def plot(self, pActivePlotter=None, default_plotting=True):
        ################################################
        ### Build Appropriate Plotter and set it up:
        #####################
        # Only Create a new BackgroundPlotter if it's needed:
        self.p = InteractiveCustomDataExplorer.build_new_plotter_if_needed(pActivePlotter, shape=self.active_config.plotting_config.subplots_shape, title=self.data_explorer_name,  plotter_type=self.active_config.plotting_config.plotter_type)
        # p.background_color = 'black'
        
        # Plot the flat arena
        if default_plotting:
            if isinstance(self.p, MultiPlotter):
                # for p in self.p:
                p = self.p[0,0] # the first plotter
                self.plots['maze_bg'] = perform_plot_flat_arena(p, self.x, self.y, bShowSequenceTraversalGradient=False)
                p.hide_axes()
                # self.p.camera_position = 'xy' # Overhead (top) view
                # apply_close_overhead_zoomed_camera_view(self.p)
                # apply_close_perspective_camera_view(self.p)
                p.render() # manually render when needed
                    
            else:
                p = self.p
                self.plots['maze_bg'] = perform_plot_flat_arena(p, self.x, self.y, bShowSequenceTraversalGradient=False)

                p.hide_axes()
                # self.p.camera_position = 'xy' # Overhead (top) view
                # apply_close_overhead_zoomed_camera_view(self.p)
                # apply_close_perspective_camera_view(self.p)
                p.render() # manually render when needed
                
        return self.p
