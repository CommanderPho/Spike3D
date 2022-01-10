#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
from pathlib import Path

from pyphocorehelpers.print_helpers import SimplePrintable


# refactored to neuropy.analyses.placefields
# class PlacefieldComputationParameters(SimplePrintable, metaclass=OrderedMeta):
#     def __init__(self, speed_thresh=3, grid_bin=2, smooth=2):
#         self.speed_thresh = speed_thresh
#         self.grid_bin = grid_bin
#         self.smooth = smooth   


class VideoOutputModeConfig(SimplePrintable):
    
    def __init__(self, active_frame_range, video_output_parent_dir, active_is_video_output_mode): 
        self.active_is_video_output_mode = active_is_video_output_mode
        self.active_frame_range = active_frame_range
  
        # Computed variables:
        if video_output_parent_dir is None:
            self.active_video_output_parent_dir = Path('output')
        else:
            self.active_video_output_parent_dir = video_output_parent_dir
            
        # active_video_output_filename = 'complete_plotting_full_curve.mp4'
        self.active_video_output_filename = 'complete_plotting_full_curve_F{}_F{}.mp4'.format(self.active_frame_range[0], self.active_frame_range[-1])
        self.active_video_output_fullpath = self.active_video_output_parent_dir.joinpath(self.active_video_output_filename)
    

class PlottingConfig(SimplePrintable):
    def __init__(self, output_subplots_shape=(1,1), output_parent_dir=None, use_age_proportional_spike_scale=False, plotter_type='BackgroundPlotter'): 
        # output_subplots_shape="3|1" means 3 plots on the left and 1 on the right,
        # output_subplots_shape="4/2" means 4 plots on top of 2 at bottom.
        # use_age_proportional_spike_scale: if True, the scale of the recent spikes is inversely proportional to their age.
        if output_subplots_shape is None:
            output_subplots_shape = (1,1) # By default, only a single plot is needed
        self.subplots_shape = output_subplots_shape
        if output_parent_dir is None:
            self.active_output_parent_dir = Path('output')
        else:
            self.active_output_parent_dir = output_parent_dir

        self.use_age_proportional_spike_scale = use_age_proportional_spike_scale
        self.plotter_type = plotter_type
        
    @property
    def figure_output_directory(self):
        return self.active_output_parent_dir     

    def get_figure_save_path(self, *args):
        # print('get_figure_save_path(...):')
        args_list = list(args)
        basename = args_list.pop()
        subdirectories = args_list
        # print(f'\tsubdirectories: {subdirectories}\n basename: {basename}')
        curr_parent_out_path = self.active_output_parent_dir.joinpath(*subdirectories)
        # print(f'\t curr_parent_out_path: {curr_parent_out_path}')
        curr_parent_out_path.mkdir(parents=True, exist_ok=True)
        return curr_parent_out_path.joinpath(basename)        

# class InteractivePlaceCellConfig:
class InteractivePlaceCellConfig(SimplePrintable):
    def __init__(self, active_session_config=None, active_epochs=None, video_output_config=None, plotting_config=None, computation_config=None):
        self.active_session_config = active_session_config
        self.active_epochs = active_epochs
        self.video_output_config = video_output_config
        self.plotting_config = plotting_config
        self.computation_config = computation_config



## For building the configs used to filter the session by epoch: 
def build_configs(session_config, active_epoch, active_subplots_shape = (1,1)):
    ## Get the config corresponding to this epoch/session settings:
    active_config = InteractivePlaceCellConfig(active_session_config=session_config, active_epochs=active_epoch, video_output_config=None, plotting_config=None) # '3|1    
    active_config.video_output_config = VideoOutputModeConfig(active_frame_range=np.arange(100.0, 120.0), 
                                                              video_output_parent_dir=Path('output', session_config.session_name, active_epoch.name),
                                                              active_is_video_output_mode=False)
    active_config.plotting_config = PlottingConfig(output_subplots_shape=active_subplots_shape,
                                                   output_parent_dir=Path('output', session_config.session_name, active_epoch.name)
                                                  )
    # Make the directories:
    active_config.plotting_config.active_output_parent_dir.mkdir(parents=True, exist_ok=True) # makes the directory if it isn't already there
    return active_config


