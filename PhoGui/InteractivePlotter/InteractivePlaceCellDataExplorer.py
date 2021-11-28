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

import PhoGui
from PhoGui.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter

from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper
from PhoGui.PhoCustomVtkWidgets import MultilineTextConsoleWidget

from PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data, build_flat_map_plot_data, perform_plot_flat_arena, build_spike_spawn_effect_light_actor, spike_geom_circle, spike_geom_box, spike_geom_cone, animal_location_circle, animal_location_trail_circle

class InteractivePlaceCellDataExplorer:
    
    def plot():
        ################################################
        ### Build Appropriate Plotter and set it up:
        #####################
        # Only Create a new BackgroundPlotter if it's needed:
        if (active_config.video_output_config.active_is_video_output_mode):
            ## Video mode should use a regular plotter object
            p = pv.Plotter(notebook=False, shape=active_config.plotting_config.subplots_shape, window_size=([1280, 720]), off_screen=True) # , line_smoothing=True, polygon_smoothing=True, multi_samples=8
        else:
            try: p
            except NameError: p = None # Checks variable p's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
            if (p is not None):
                if isinstance(p, pvqt.BackgroundPlotter):
                    if p.app_window.isHidden():
                        print('No open BackgroundPlotter')
                        p.close() # Close it to start over fresh
                        p = None
                        needs_create_new_backgroundPlotter = True
                    else:
                        print('BackgroundPlotter already open, reusing it.. NOT Forcing creation of a new one!')
                        # p.app_window.window().show()
                        # p.clear()
                        # needs_create_new_backgroundPlotter = False                
                        p.close() # Close it to start over fresh
                        p = None
                        needs_create_new_backgroundPlotter = True
                        
                else:
                    print('No open BackgroundPlotter, p is a Plotter object')
                    p.close()
                    p = None
                    needs_create_new_backgroundPlotter = True
            else:
                print('No extant BackgroundPlotter')
                needs_create_new_backgroundPlotter = True
            if needs_create_new_backgroundPlotter:
                print('Creating a new BackgroundPlotter')
                p = pvqt.BackgroundPlotter(window_size=(1920, 1080), shape=active_config.plotting_config.subplots_shape, off_screen=False) # Use just like you would a pv.Plotter() instance
                print('done.')

        # p.background_color = 'black'

        if (not active_config.video_output_config.active_is_video_output_mode):
            #Interactive Mode: Enable interactive controls:
            interactive_timestamp_slider_actor = p.add_slider_widget(on_slider_update_mesh, [0, (num_time_points-1)], title='Trajectory Timestep', event_type='always', style='modern', pointa=(0.025, 0.1), pointb=(0.98, 0.1), fmt='%0.2f') # fmt="%0.2f"
            # interactive_timestamp_slider_wrapper = InteractiveSliderWrapper(interactive_timestamp_slider_actor)    
            # interactive_plotter = PhoGui.InteractivePlotter.PhoInteractivePlotter.PhoInteractivePlotter(pyvista_plotter=p, interactive_timestamp_slider_actor=interactive_timestamp_slider_actor)
            interactive_plotter = PhoInteractivePlotter(pyvista_plotter=p, interactive_timestamp_slider_actor=interactive_timestamp_slider_actor)
            # interactive_checkbox_actor = p.add_checkbox_button_widget(toggle_animation, value=False, color_on='green')
            helper_controls_text = print_controls_helper_text()
            p.add_text(helper_controls_text, position='upper_left', name='lblControlsHelperText', color='grey', font_size=8.0)
            
            
        # Plot the flat arena
        perform_plot_flat_arena(p, x, y, bShowSequenceTraversalGradient=False)
        # Legend:
        legend_entries = [['pf[{}]'.format(active_epoch_session.neuron_ids[i]), pf_colors[:,i]] for i in np.arange(len(active_epoch_session.neuron_ids))]
        if show_legend:
            legendActor = p.add_legend(legend_entries, name='interactiveSpikesPositionLegend', 
                                    bcolor=(0.05, 0.05, 0.05), border=True,
                                    origin=[0.95, 0.3], size=[0.05, 0.65]) # vtk.vtkLegendBoxActor
        else:
            legendActor = None

        # p.show_grid()
        # p.add_axes(line_width=5, labels_off=True)
        p.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0) # Supposedly helps with translucency
        p.hide_axes()
        # p.camera_position = 'xy' # Overhead (top) view
        # apply_close_overhead_zoomed_camera_view(p)
        # apply_close_perspective_camera_view(p)
        p.render() # manually render when needed

        if active_config.video_output_config.active_is_video_output_mode:
            active_config.video_output_config.active_video_output_parent_dir.mkdir(parents=True, exist_ok=True) # makes the directory if it isn't already there
            print('Writing video to {}...'.format(active_config.video_output_config.active_video_output_fullpath))
            p.show(auto_close=False)
            make_mp4_from_plotter(p, active_config.video_output_config.active_frame_range, on_slider_update_mesh, filename=active_config.video_output_config.active_video_output_fullpath, framerate=60) # 60fps
            p.close()
            p = None

        # p.show()
                        
        print('all done!')
