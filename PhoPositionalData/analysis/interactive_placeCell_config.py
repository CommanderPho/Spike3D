#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
from pathlib import Path

from PhoGui.general_helpers import OrderedMeta, SimplePrintable

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


def print_subsession_neuron_differences(prev_session_Neurons, subsession_Neurons):
    num_original_neurons = prev_session_Neurons.n_neurons
    num_original_total_spikes = np.sum(prev_session_Neurons.n_spikes)
    num_subsession_neurons = subsession_Neurons.n_neurons
    num_subsession_total_spikes = np.sum(subsession_Neurons.n_spikes)
    print('{}/{} total spikes spanning {}/{} units remain in subsession'.format(num_subsession_total_spikes, num_original_total_spikes, num_subsession_neurons, num_original_neurons))
