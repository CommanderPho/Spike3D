#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""


class InterfaceProperties(object):
    """ Holds user interface state, such as the current animation status or the slider's values """
    def __init__(self, active_timestamp_slider_wrapper):
        # self.curr_plot_update_step = 1 # Update every frame
        # self.curr_plot_update_frequency = self.curr_plot_update_step * active_epoch_pos.sampling_rate # number of updates per second (Hz)
        # self.num_time_points = active_epoch_pos.n_frames / self.curr_plot_update_step        
        # self.position_trail_max_duration = 0
        # self.max_historical_spikes_age = 0 # How long ago the historical spikes could have been plotted and still not be removed.
        self.active_timestamp_slider_wrapper = active_timestamp_slider_wrapper # Used to actually update the slider when it's appropriate
        self.animation_state = False # Whether it's playing or not

    def __call__(self):
        if self.animation_state:
            # only if animation is currently active:
            self.active_timestamp_slider_wrapper.step_index(15) # TODO: allow variable step size
            pass
        
        