import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class VisualizationWindow(object):
    """Docstring for VisualizationWindow."""
    duration_seconds: float = None
    sampling_rate: float = None
    duration_num_frames: int = None

    def __init__(self, duration_seconds=None, sampling_rate=None, duration_num_frames=None):
        self.duration_seconds = duration_seconds # Update every frame
        if (sampling_rate is not None):
            self.sampling_rate = sampling_rate # number of updates per second (Hz)
        else:
            print('Sampling rate is none!')
            self.sampling_rate = None
            
        if (duration_num_frames is not None):
            self.duration_num_frames = duration_num_frames
        else:
            self.duration_num_frames = VisualizationWindow.compute_window_samples(self.duration_seconds, self.sampling_rate)

    def build_sliding_windows(self, times):
        return VisualizationWindow.compute_sliding_windows(times, self.duration_num_frames)


    @staticmethod
    def compute_window_samples(window_duration_seconds, sampling_rate):
        return int(np.floor(window_duration_seconds * sampling_rate))
        
    @staticmethod
    def compute_sliding_windows(times, num_window_frames):
        # build a sliding window to be able to retreive the correct flattened indicies for any given timestep
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(times, num_window_frames)

    
