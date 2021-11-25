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

# from PhoPositionalData.plotting.animations import * # make_mp4_from_plotter

## For animation/movie creation, see:
# https://github.com/pyvista/pyvista-support/issues/81

## Save out to MP4 Movie
def make_mp4_from_plotter(active_plotter, active_frame_range, update_callback, filename='sphere-shrinking.mp4', framerate=30):
    # Open a movie file
    print('active_frame_range: {}'.format(active_frame_range))
    try:
        # Further file processing goes here
        print('Trying to open mp4 movie file at {}...\n'.format(filename))
        if isinstance(active_plotter, pv.plotting.Plotter):
            active_plotter.show(auto_close=False)  # only necessary for an off-screen movie
        elif isinstance(active_plotter, pvqt.BackgroundPlotter):
            active_plotter.show()
        else:
            print('ERROR: active_plotter is not a Plotter or a BackgroundPlotter! Is it valid?')
            
        active_plotter.open_movie(filename, framerate=framerate)
        # Run through each frame
        active_plotter.write_frame()  # write initial data
        total_number_frames = np.size(active_frame_range)
        print('\t opened. Planning to write {} frames...\n'.format(total_number_frames))
        # Update scalars on each frame
        for i in active_frame_range:
            # print('\t Frame[{} of {}]'.format(i, total_number_frames))
            # Call the provided update_callback function:
            update_callback(i)
            # active_plotter.add_text(f"Iteration: {i}", name='time-label')
            # active_plotter.render()
            active_plotter.write_frame()  # Write this frame

    finally:
        # Be sure to close the plotter when finished
        active_plotter.close()
        print('File reader closed!')
        
    print('done.')
    
    
###########################    
## Batched/Ghosting Plotting Methods:

### UNUSED
def test_on_time_update_mesh(currTime):
    print('main_spikes_mesh.array_names: {}\n shape of main_spikes_mesh[times]: {}'.format(main_spikes_mesh.array_names, np.shape(main_spikes_mesh['times'])))
    print('currTime: {}'.format(currTime))
    curr_ghosts = np.argwhere(main_spikes_mesh['times'] > currTime) # any times greater than the currTime
    print('shape of curr_ghosts: {}'.format(np.shape(curr_ghosts)))
    # This will act on the mesh inplace to mark those cell indices as ghosts
    # main_spikes_mesh.remove_cells(curr_ghosts)
    thresholded_main_spikes_mesh = main_spikes_mesh.threshold(value=(0.0, currTime), scalars='times', continuous=True, preference='point')
    # thresholded_main_spikes_mesh.plot(cmap='gist_earth_r', show_scalar_bar=False, show_edges=True)
    print('thresholded_main_spikes_mesh: {}'.format(thresholded_main_spikes_mesh))
    if thresholded_main_spikes_mesh.n_points >= 1:
            # main_spikes_mesh_actor = p.add_mesh(spikes_pc, name='spikes_main', scalars='cellID', cmap=active_cells_listed_colormap, show_scalar_bar=False, render=False)
            main_spikes_mesh_actor = p.add_mesh(thresholded_main_spikes_mesh, name='spikes_main', scalars='cellID', cmap=active_cells_listed_colormap, show_scalar_bar=False, render=False)

### UNUSED
def test_batch_plot_all_spikes():
    # plots all the spikes at once but sets them invisible, revealing them as needed
    active_included_indicies = np.isfinite(flattened_spikes.flattened_spike_times) # Accumulate Spikes mode. All spikes occuring prior to the end of the frame (meaning the current time) are plotted
    active_flattened_spike_times = flattened_spikes.flattened_spike_times[active_included_indicies]
    # active_flattened_spike_identities = flattened_spikes.flattened_spike_identities[active_included_indicies] # actual UnitID is the identity for each spike
    active_flattened_spike_identities = flattened_spike_active_unitIdentities[active_included_indicies] # a relative index starting at 0 and going up to the number of active units is the identity for each spike
    active_flattened_spike_positions_list = flattened_spike_positions_list[:, active_included_indicies]
    spikes_pdata, spikes_pc = build_active_spikes_plot_data(active_flattened_spike_times, active_flattened_spike_identities, active_flattened_spike_positions_list)
    spikes_pc_grid_mesh = spikes_pc.cast_to_unstructured_grid()
    # main_spikes_mesh = p.add_mesh(spikes_pc, name='spikes_main', scalars='cellID', cmap='rainbow', show_scalar_bar=True) # , color=active_cells_colormap[original_cell_id]
    if spikes_pc.n_points >= 1:
        # main_spikes_mesh_actor = p.add_mesh(spikes_pc, name='spikes_main', scalars='cellID', cmap=active_cells_listed_colormap, show_scalar_bar=False, render=False)
        main_spikes_mesh_actor = p.add_mesh(spikes_pc_grid_mesh, name='spikes_main', scalars='cellID', cmap=active_cells_listed_colormap, show_scalar_bar=False, render=False)
        
    # main_spikes_mesh = main_spikes_mesh.cast_to_unstructured_grid()
    return spikes_pc_grid_mesh, main_spikes_mesh_actor


# ### UNUSED
# if should_use_test_batch_plotting_methods:
#     # main_spikes_mesh = test_batch_plot_all_spikes()
#     main_spikes_mesh, main_spikes_mesh_actor = test_batch_plot_all_spikes()
    

