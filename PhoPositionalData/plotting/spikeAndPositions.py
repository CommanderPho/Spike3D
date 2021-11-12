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


# Fixed:
animal_location_sphere = pv.Sphere(radius=2.3)
animal_location_direction_cone = pv.Cone()
animal_location_circle = pv.Circle(radius=8.0)
animal_location_trail_circle = pv.Circle(radius=2.3)

## Spike indicator geometry:
spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2)
spike_geom_circle = pv.Circle(radius=0.4)


# Call with:
# pdata_maze, pc_maze = build_flat_map_plot_data() # Plot the flat arena
# p.add_mesh(pc_maze, name='maze_bg', color="black", render=False)
def build_flat_map_plot_data():
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
