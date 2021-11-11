import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class VisualizationWindow(object):
    """Docstring for VisualizationWindow."""
    duration_seconds: float
    duration_num_frames: int
    
    def build_sliding_windows(self, times):
       return VisualizationWindow.compute_sliding_windows(times, self.duration_num_frames)

    @staticmethod
    def compute_sliding_windows(times, num_window_frames):
        # build a sliding window to be able to retreive the correct flattened indicies for any given timestep
        from numpy.lib.stride_tricks import sliding_window_view
        return sliding_window_view(times, num_window_frames)
    
    
