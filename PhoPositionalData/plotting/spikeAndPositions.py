#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
from pathlib import Path


# Fixed Geometry objects:
animal_location_sphere = pv.Sphere(radius=2.3)
animal_location_direction_cone = pv.Cone()
animal_location_circle = pv.Circle(radius=8.0)
animal_location_trail_circle = pv.Circle(radius=2.3)

## Spike indicator geometry:
spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
spike_geom_circle = pv.Circle(radius=0.4)
spike_geom_box = pv.Box(bounds=[-0.2, 0.2, -0.2, 0.2, -0.05, 0.05])
# pv.Cylinder


# Call with:
# pdata_maze, pc_maze = build_flat_map_plot_data() # Plot the flat arena
# p.add_mesh(pc_maze, name='maze_bg', color="black", render=False)
def build_flat_map_plot_data(x, y):
    # Builds the flat base maze map that the other data will be plot on top of
    ## Implicitly relies on: x, y
    z = np.zeros_like(x)
    point_cloud = np.vstack((x, y, z)).T
    pdata = pv.PolyData(point_cloud)
    pdata['occupancy heatmap'] = np.arange(np.shape(point_cloud)[0])
    geo = pv.Circle(radius=0.5)
    pc = pdata.glyph(scale=False, geom=geo)
    return pdata, pc


# TODO: brought in from old file, finish implementation
def build_active_spikes_plot_data(active_flattened_spike_times, active_flattened_spike_identities, active_flattened_spike_positions_list, spike_geom):
    spike_series_times = active_flattened_spike_times # currently unused
    spike_series_identities = active_flattened_spike_identities # currently unused
    spike_series_positions = active_flattened_spike_positions_list
    # z = np.zeros_like(spike_series_positions[0,:])
    z_fixed = np.full_like(spike_series_positions[0,:], 1.1) # Offset a little bit in the z-direction so we can see it
    spike_history_point_cloud = np.vstack((spike_series_positions[0,:], spike_series_positions[1,:], z_fixed)).T
    spike_history_pdata = pv.PolyData(spike_history_point_cloud)
    # spike_history_pdata['times'] = spike_series_times
    spike_history_pdata['cellID'] = spike_series_identities
    # create many spheres from the point cloud
    spike_history_pc = spike_history_pdata.glyph(scale=False, geom=spike_geom.copy())
    return spike_history_pdata, spike_history_pc


## This light effect occurs when a spike happens to indicate its presence
light_spawn_constant_z_offset = 2.5
light_spawn_constant_z_focal_position = -0.5 # by default, the light focuses under the floor

def build_spike_spawn_effect_light_actor(p, spike_position, spike_unit_color='white'):
    # spike_position: should be a tuple like (0, 0, 10)
    light_source_position = spike_position
    light_source_position[3] = light_source_position[3] + light_spawn_constant_z_offset
    light_focal_point = spike_position
    light_focal_point[3] = light_focal_point[3] + light_spawn_constant_z_focal_position
    
    SpikeSpawnEffectLight = pv.Light(position=light_source_position, focal_point=light_focal_point, color=spike_unit_color)
    SpikeSpawnEffectLight.positional = True
    SpikeSpawnEffectLight.cone_angle = 40
    SpikeSpawnEffectLight.exponent = 10
    SpikeSpawnEffectLight.intensity = 3
    SpikeSpawnEffectLight.show_actor()
    p.add_light(SpikeSpawnEffectLight)
    return SpikeSpawnEffectLight # return the light actor for removal later



class InteractiveSliderWrapper:

    # instance attributes
    def __init__(self, slider_obj):
        self.slider_obj = slider_obj
        # curr_min_value = self.slider_obj.GetRepresentation().GetMinimumValue()
        # curr_max_value = self.slider_obj.GetRepresentation().GetMaximumValue()
        # curr_value = self.slider_obj.GetRepresentation().GetValue()

    @property
    def curr_value(self):
        """The curr_value property."""
        return self.slider_obj.GetRepresentation().GetValue()
    
    @curr_value.setter
    def curr_value(self, value):
        self.update_value(value)


    @property
    def curr_index(self):
        """The curr_index property."""
        return int(self.curr_value)
    
    @curr_index.setter
    def curr_index(self, value):
        self.update_value(float(value))


    # instance method
    def update_value(self, new_value):
        from pyvista import _vtk
        self.slider_obj.GetRepresentation().SetValue(new_value)
        self.slider_obj.InvokeEvent(_vtk.vtkCommand.InteractionEvent) # Called to ensure user callback is performed.


    def _safe_constrain_index(self, proposed_index):
        curr_min_value = self.slider_obj.GetRepresentation().GetMinimumValue()
        curr_max_value = self.slider_obj.GetRepresentation().GetMaximumValue()
        
        if (proposed_index < curr_min_value):
            print('value too low!')
            proposed_index = curr_min_value
        elif (curr_max_value < proposed_index):
            print('value too high!')
            proposed_index = curr_max_value
        else:
            # proposed_index is within the range and is fine
            pass
        return proposed_index
    
        
    def step_index(self, step_size):
        curr_index = self.curr_index
        proposed_index = self._safe_constrain_index(curr_index + step_size)
        self.curr_index = proposed_index
        return proposed_index
    
    def step_next_index(self):
        return self.step_index(1)
        
    def step_prev_index(self):
        return self.step_index(-1) 

