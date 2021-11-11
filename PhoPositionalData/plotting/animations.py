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