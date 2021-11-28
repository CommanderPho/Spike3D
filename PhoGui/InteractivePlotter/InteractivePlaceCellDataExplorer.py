#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho


"""
import numpy as np
import pyvista as pv


from PhoPositionalData.plotting.spikeAndPositions import plot_placefields2D, update_plotVisiblePlacefields2D, build_custom_placefield_maps_lookup_table
from PhoPositionalData.plotting.gui import SetVisibilityCallback, MutuallyExclusiveRadioButtonGroup, add_placemap_toggle_checkboxes, add_placemap_toggle_mutually_exclusive_checkboxes
from PhoPositionalData.plotting.animations import make_mp4_from_plotter

from PhoGui.InteractivePlotter.PhoInteractivePlotter import PhoInteractivePlotter

from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper
from PhoGui.PhoCustomVtkWidgets import MultilineTextConsoleWidget

from PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data, build_flat_map_plot_data, perform_plot_flat_arena, build_spike_spawn_effect_light_actor, spike_geom_circle, spike_geom_box, spike_geom_cone, animal_location_circle, animal_location_trail_circle

from PhoGui.InteractivePlotter.shared_helpers import InteractivePyvistaPlotterBuildIfNeededMixin

class InteractivePlaceCellDataExplorer(InteractivePyvistaPlotterBuildIfNeededMixin):
# show_legend = True

def __init__(self, active_config, active_session, t, x, y, num_time_points, extant_plotter=None):
    self.active_config = active_config
    self.active_session = active_session
    # self.active_config = self.active_session.config
    self.t = t
    self.x = x
    self.y = y
    
    # active_epoch_session_Neurons, active_epoch_pos, active_epoch_position_times = self.active_session.neurons, self.active_session.position, self.active_session.position.time
    # Position variables: t, x, y
    t = self.active_session.position.time
    x = self.active_session.position.x
    y = self.active_session.position.y
    linear_pos = self.active_session.position.linear_pos
    speeds = self.active_session.position.speed 

    
    
    ## for plot(...):
    # pf_colors, active_config
    
    ## for on_slider_update_mesh(...):
    # pre_computed_window_sample_indicies, longer_spikes_window,
    # flattened_spikes.flattened_spike_times, flattened_spike_active_unitIdentities, flattened_spike_positions_list,
    # active_cells_listed_colormap
    # recent_spikes_window
    # z_fixed, 
    # active_trail_opacity_values, active_trail_size_values



#         self.active_epoch_session = active_epoch_session
    
#         self.pre_computed_window_sample_indicies = pre_computed_window_sample_indicies
#         self.active_window_sample_indicies = active_window_sample_indicies
#         self.flattened_spikes = flattened_spikes
#         self.flattened_spike_active_unitIdentities = flattened_spike_active_unitIdentities
#         self.flattened_spike_positions_list = flattened_spike_positions_list
#         self.active_cells_listed_colormap = active_cells_listed_colormap
    
#         self.active_trail_opacity_values = active_trail_opacity_values
#         self.active_trail_size_values = active_trail_size_values
    
    self.num_time_points = num_time_points
    # curr_min_value = self.slider_obj.GetRepresentation().GetMinimumValue()
    # curr_max_value = self.slider_obj.GetRepresentation().GetMaximumValue()
    # curr_value = self.slider_obj.GetRepresentation().GetValue()
    self.p = extant_plotter
    
######################
# General Plotting Method:    
# pre_computed_window_sample_indicies, longer_spikes_window,
# flattened_spikes.flattened_spike_times, flattened_spike_active_unitIdentities, flattened_spike_positions_list,
# active_cells_listed_colormap
# recent_spikes_window
# z_fixed, 
# active_trail_opacity_values, active_trail_size_values
def on_slider_update_mesh(self, value):
    curr_i = int(value)    
    active_window_sample_indicies = np.squeeze(pre_computed_window_sample_indicies[curr_i,:]) # Get the current precomputed indicies for this curr_i

    ## Spike Plotting:
    # Get the times that fall within the current plot window:
    curr_time_fixedSegments = t[active_window_sample_indicies] # New Way
    t_start = curr_time_fixedSegments[0]
    t_stop = curr_time_fixedSegments[-1]
    # print('Constraining to curr_time_fixedSegments with times (start: {}, end: {})'.format(t_start, t_stop))
    # print('curr_time_fixedSegments: {}'.format(curr_time_fixedSegments))
    curr_text_rendering_string = 'curr_i: {:d}; (t_start: {:.2f}, t_stop: {:.2f})'.format(curr_i, t_start, t_stop) # :.3f
    self.p.add_text(curr_text_rendering_string, name='lblCurrent_spike_range', position='lower_right', color='white', shadow=True, font_size=10)

    ## Historical Spikes:
    # active_included_all_historical_indicies = (flattened_spikes.flattened_spike_times < t_stop) # Accumulate Spikes mode. All spikes occuring prior to the end of the frame (meaning the current time) are plotted
    historical_t_start = (t_stop - longer_spikes_window.duration_seconds) # Get the earliest time that will be included in the search
    active_included_all_historical_indicies = ((flattened_spikes.flattened_spike_times > historical_t_start) & (flattened_spikes.flattened_spike_times < t_stop)) # Two Sided Range Mode
    historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data(flattened_spikes.flattened_spike_times[active_included_all_historical_indicies],
                                                                                    flattened_spike_active_unitIdentities[active_included_all_historical_indicies],
                                                                                    flattened_spike_positions_list[:, active_included_all_historical_indicies],
                                                                                    spike_geom=spike_geom_box.copy())
    if historical_spikes_pc.n_points >= 1:
        historical_main_spikes_mesh = self.p.add_mesh(historical_spikes_pc, name='historical_spikes_main', scalars='cellID', cmap=active_cells_listed_colormap, show_scalar_bar=False, lighting=True, render=False)

    ## Actively Firing Spikes:
    recent_spikes_t_start = (t_stop - recent_spikes_window.duration_seconds) # Get the earliest time that will be included in the recent spikes
    # print('recent_spikes_t_start: {}; t_start: {}'.format(recent_spikes_t_start, t_start))
    active_included_recent_only_indicies = ((flattened_spikes.flattened_spike_times > recent_spikes_t_start) & (flattened_spikes.flattened_spike_times < t_stop)) # Two Sided Range Mode
    # active_included_recent_only_indicies = ((flattened_spikes.flattened_spike_times > t_start) & (flattened_spikes.flattened_spike_times < t_stop)) # Two Sided Range Mode
    recent_only_spikes_pdata, recent_only_spikes_pc = build_active_spikes_plot_data(flattened_spikes.flattened_spike_times[active_included_recent_only_indicies],
                                                                                    flattened_spike_active_unitIdentities[active_included_recent_only_indicies],
                                                                                    flattened_spike_positions_list[:, active_included_recent_only_indicies],
                                                                                    spike_geom=spike_geom_cone.copy())
    if recent_only_spikes_pc.n_points >= 1:
        recent_only_main_spikes_mesh = self.p.add_mesh(recent_only_spikes_pc, name='recent_only_spikes_main', scalars='cellID', cmap=active_cells_listed_colormap, show_scalar_bar=False, lighting=False, render=False) # color='white'

    ## Animal Position and Location Trail Plotting:
    point_cloud_fixedSegements_positionTrail = np.column_stack((self.x[active_window_sample_indicies], self.y[active_window_sample_indicies], z_fixed))
    pdata_positionTrail = pv.PolyData(point_cloud_fixedSegements_positionTrail.copy()) # a mesh
    pdata_positionTrail.point_data['pho_fade_values'] = active_trail_opacity_values
    pdata_positionTrail.point_data['pho_size_values'] = active_trail_size_values
    # create many spheres from the point cloud
    pc_positionTrail = pdata_positionTrail.glyph(scale='pho_size_values', geom=animal_location_trail_circle)
    animal_location_trail_mesh = self.p.add_mesh(pc_positionTrail, name='animal_location_trail', ambient=0.6, opacity='linear_r', scalars='pho_fade_values', nan_opacity=0.0,
                                            show_edges=False, render_lines_as_tubes=True, show_scalar_bar=False, use_transparency=True, render=False) # works to render a heat colored (most recent==hotter) position

    ## Animal Current Position:
    curr_animal_point = point_cloud_fixedSegements_positionTrail[-1,:].copy() # Get the last point
    pdata_current_point = pv.PolyData(curr_animal_point) # a mesh
    pc_current_point = pdata_current_point.glyph(scale=False, geom=animal_location_circle)
    animal_current_location_point_mesh = self.p.add_mesh(pc_current_point, name='animal_location', color='green', ambient=0.6, opacity=0.5,
                                                    show_edges=True, edge_color=[0.05, 0.8, 0.08], line_width=3.0, nan_opacity=0.0, render_lines_as_tubes=True,
                                                    show_scalar_bar=False, use_transparency=True, render=False) # works to render a heat colored (most recent==hotter) position

    self.p.render() # renders to ensure it's updated after changing the ScalarVisibility above
    # self.p.update()
    # self.p.app.processEvents() # not needed probably
    return

# curr_active_neuron_pf_identifier = 'pf[{}]'.format(curr_active_neuron_ID)
# label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier


# pf_colors, active_config
def plot(self, pActivePlotter=None):
    ################################################
    ### Build Appropriate Plotter and set it up:
    #####################
    # Only Create a new BackgroundPlotter if it's needed:
    if (self.active_config.video_output_config.active_is_video_output_mode):
        ## Video mode should use a regular plotter object
        self.p = pv.Plotter(notebook=False, shape=self.active_config.plotting_config.subplots_shape, window_size=([1280, 720]), off_screen=True) # , line_smoothing=True, polygon_smoothing=True, multi_samples=8
    else:
        self.p = InteractivePlaceCellDataExplorer.build_new_plotter_if_needed(pActivePlotter, shape=self.active_config.plotting_config.subplots_shape)

    # p.background_color = 'black'

    if (not self.active_config.video_output_config.active_is_video_output_mode):
        #Interactive Mode: Enable interactive controls:
        interactive_timestamp_slider_actor = self.p.add_slider_widget(self.on_slider_update_mesh, [0, (self.num_time_points-1)], title='Trajectory Timestep', event_type='always', style='modern', pointa=(0.025, 0.1), pointb=(0.98, 0.1), fmt='%0.2f') # fmt="%0.2f"
        # interactive_timestamp_slider_wrapper = InteractiveSliderWrapper(interactive_timestamp_slider_actor)    
        # interactive_plotter = PhoGui.InteractivePlotter.PhoInteractivePlotter.PhoInteractivePlotter(pyvista_plotter=p, interactive_timestamp_slider_actor=interactive_timestamp_slider_actor)
        interactive_plotter = PhoInteractivePlotter(pyvista_plotter=self.p, interactive_timestamp_slider_actor=interactive_timestamp_slider_actor)
        # interactive_checkbox_actor = p.add_checkbox_button_widget(toggle_animation, value=False, color_on='green')
        helper_controls_text = print_controls_helper_text()
        self.p.add_text(helper_controls_text, position='upper_left', name='lblControlsHelperText', color='grey', font_size=8.0)
        
    # Plot the flat arena
    perform_plot_flat_arena(self.p, self.x, self.y, bShowSequenceTraversalGradient=False)
    
    # Legend:
    legend_entries = [['pf[{}]'.format(self.active_session.neuron_ids[i]), pf_colors[:,i]] for i in np.arange(len(self.active_session.neuron_ids))]
    if self.active_config.plotting_config.show_legend:
        legendActor = self.p.add_legend(legend_entries, name='interactiveSpikesPositionLegend', 
                                    bcolor=(0.05, 0.05, 0.05), border=True,
                                    origin=[0.95, 0.3], size=[0.05, 0.65]) # vtk.vtkLegendBoxActor
    else:
        legendActor = None

    
    self.p.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0) # Supposedly helps with translucency
    self.p.hide_axes()
    # self.p.camera_position = 'xy' # Overhead (top) view
    # apply_close_overhead_zoomed_camera_view(self.p)
    # apply_close_perspective_camera_view(self.p)
    self.p.render() # manually render when needed

    if self.active_config.video_output_config.active_is_video_output_mode:
        self.active_config.video_output_config.active_video_output_parent_dir.mkdir(parents=True, exist_ok=True) # makes the directory if it isn't already there
        print('Writing video to {}...'.format(self.active_config.video_output_config.active_video_output_fullpath))
        self.p.show(auto_close=False)
        make_mp4_from_plotter(self.p, self.active_config.video_output_config.active_frame_range, self.on_slider_update_mesh, filename=self.active_config.video_output_config.active_video_output_fullpath, framerate=60) # 60fps
        self.p.close()
        self.p = None
        
    return self.p
