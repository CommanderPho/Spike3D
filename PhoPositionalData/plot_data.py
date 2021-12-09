#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
from threading import Thread
from ipygany import PolyMesh, Scene, IsoColor, WarpByScalar
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
from pathlib import Path
import bqplot.scales
import seaborn as sns
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# import mplcursors
import math # For color map generation
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

import ipywidgets as widgets

from PhoPositionalData.process_data import get_filtered_window


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

# Plot current segment as spline:
def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

# Plot raster plot:
# Set different colors for each neuron
def plot_raster_plot(spike_list, spike_positions_list):
    min_timestep = 0
    max_timestep = 4000
    active_spike_indices, active_spike_list, active_spike_positions_list = get_filtered_window(spike_list, spike_positions_list, min_timestep=0, max_timestep=400)

    # cmap = get_cmap(num_cells)
    cmap = generate_colormap(num_cells)

    plt.figure()
    for i, cell_spike_positions in enumerate(spike_positions_list):
       plt.scatter(cell_spike_positions[0,:], cell_spike_positions[1,:], s=0.1, color=cmap(i), alpha=0.2)

    plt.title('Spike positions')
    plt.show()

def plot_eventplot_raster_plot(active_spike_list, cmap):
    num_cells = len(active_spike_list)
    colorCodes = np.broadcast_to(np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                            [1, 0, 1]]), (num_cells, 3))
    # print(np.shape(colorCodes))                    
    # Set spike colors for each neuron
    lineSize = [0.4, 0.3, 0.2, 0.8, 0.5, 0.6, 0.7, 0.9]                                  

    # Draw a spike raster plot
    # plt.eventplot(spike_cells[0])
    colorCodes = [cmap(i) for i in np.arange(num_cells)]
    plt.eventplot(active_spike_list, color=colorCodes)
    # plt.eventplot(spike_matrix)    
    # plt.eventplot(spike_matrix, color=colorCodes, linelengths = lineSize)     
    # Provide the title for the spike raster plot
    plt.title('Spike raster plot')
    # Give x axis label for the spike raster plot
    plt.xlabel('Neuron')
    # Give y axis label for the spike raster plot
    plt.ylabel('Spike')
    # Display the spike raster plot
    plt.show()