#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
from pathlib import Path

class VideoOutputModeConfig:
    
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
    

class PlottingConfig:
    def __init__(self, output_subplots_shape=(1,1), output_parent_dir=None, use_age_proportional_spike_scale=False): 
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
        


class InteractivePlaceCellConfig:
    
    # class attribute
    # species = 'bird'

    # instance attribute
    def __init__(self, active_epochs, video_output_config, plotting_config):
        self.active_epochs = active_epochs
        self.video_output_config = video_output_config
        self.plotting_config = plotting_config



