#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
from pathlib import Path

from PhoGui.general_helpers import OrderedMeta

class NamedEpoch(metaclass=OrderedMeta):
    def __init__(self, name, start_end_times):
        self.name = name
        self.start_end_times = start_end_times
        
        
class SessionConfig(metaclass=OrderedMeta):
    def __init__(self, basepath, session_name=None):
        """[summary]
        Args:
            active_epoch (NamedEpoch): [description]
            basepath (Path): [description].
            session_name (str, optional): [description].
        """
        self.basepath = basepath
        if session_name is None:
            session_name = Path(basepath).parts[-1]
        self.session_name = session_name
        # self.active_epoch = active_epoch

class PlacefieldComputationParameters(metaclass=OrderedMeta):
    def __init__(self, speed_thresh=3, grid_bin=2, smooth=2):
        self.speed_thresh = speed_thresh
        self.grid_bin = grid_bin
        self.smooth = smooth   


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
        


# class InteractivePlaceCellConfig:
class InteractivePlaceCellConfig:
    def __init__(self, active_session_config=None, active_epochs=None, video_output_config=None, plotting_config=None, computation_config=None):
        self.active_session_config = active_session_config
        self.active_epochs = active_epochs
        self.video_output_config = video_output_config
        self.plotting_config = plotting_config
        self.computation_config = computation_config




def get_subsession_for_epoch(sess, active_epoch_name, active_epoch_times):
    # active_config = InteractivePlaceCellConfig(active_epoch_name,
    #                     VideoOutputModeConfig(active_frame_range=np.arange(11070.0, 13970.0), video_output_parent_dir=Path('output', session_name, active_epoch_name), active_is_video_output_mode=False),
    #                     PlottingConfig(output_subplots_shape=active_subplots_shape, output_parent_dir=Path('output', session_name, active_epoch_name))) # '3|1     
    print('Constraining to epoch with times (start: {}, end: {})'.format(active_epoch_times[0], active_epoch_times[1]))
    # (start: 603785449121.0, end: 603785451229.5245); (start: 603,785,449,121.0, end: 603,785,451,229.5245)
    # active_epoch_session_Neurons = sess.neurons.get_neuron_type('pyramidal')
    # active_epoch_session_Neurons = sess.neurons.time_slice(active_epoch_times[0], active_epoch_times[1]) # Filter by pyramidal cells only, returns a core.Neurons object with its spiketrains filtered for the provided start/end times
    active_epoch_session_Neurons = sess.neurons.get_neuron_type('pyramidal').time_slice(active_epoch_times[0], active_epoch_times[1]) # Filter by pyramidal cells only, returns a core.Neurons object with its spiketrains filtered for the provided start/end times
    active_epoch_position_times_index_mask = sess.position.time_slice_indicies(active_epoch_times[0], active_epoch_times[1]) # a Boolean selection mask
    active_epoch_position_times = sess.position.time[active_epoch_position_times_index_mask] # The actual times
    active_epoch_relative_position_times = active_epoch_position_times - active_epoch_position_times[0] # Subtract off the first index, so that it becomes zero
    active_epoch_pos = sess.position.time_slice(active_epoch_times[0], active_epoch_times[1]) # active_epoch_pos's .time and start/end are all valid
    # have active_epoch_position_times: the actual times each position sample occured in seconds, active_epoch_relative_position_times: the same as active_epoch_position_times but starting at zero. Finally, have a complete active_epoch_pos object
    print_subsession_neuron_differences(sess.neurons, active_epoch_session_Neurons)
    return active_epoch_session_Neurons, active_epoch_pos



def print_subsession_neuron_differences(prev_session_Neurons, subsession_Neurons):
    num_original_neurons = prev_session_Neurons.n_neurons
    num_original_total_spikes = np.sum(prev_session_Neurons.n_spikes)
    num_subsession_neurons = subsession_Neurons.n_neurons
    num_subsession_total_spikes = np.sum(subsession_Neurons.n_spikes)
    print('{}/{} total spikes spanning {}/{} units remain in subsession'.format(num_subsession_total_spikes, num_original_total_spikes, num_subsession_neurons, num_original_neurons))
