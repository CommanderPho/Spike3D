#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def process_positionalAnalysis_data(data):
    t = np.squeeze(data['positionalAnalysis']['track_position']['t'])
    x = np.squeeze(data['positionalAnalysis']['track_position']['x'])
    y = np.squeeze(data['positionalAnalysis']['track_position']['y'])
    speeds = np.squeeze(data['positionalAnalysis']['track_position']['speeds'])
    dt = np.squeeze(data['positionalAnalysis']['displacement']['dt'])
    dx = np.squeeze(data['positionalAnalysis']['displacement']['dx'])
    dy = np.squeeze(data['positionalAnalysis']['displacement']['dy'])
    return t,x,y,speeds,dt,dx,dy


#todo:
def process_finalSpikingDatasitionalAnalysis_data(data):
    t = np.squeeze(data['positionalAnalysis']['track_position']['t'])
    x = np.squeeze(data['positionalAnalysis']['track_position']['x'])
    y = np.squeeze(data['positionalAnalysis']['track_position']['y'])
    speeds = np.squeeze(data['positionalAnalysis']['track_position']['speeds'])
    dt = np.squeeze(data['positionalAnalysis']['displacement']['dt'])
    dx = np.squeeze(data['positionalAnalysis']['displacement']['dx'])
    dy = np.squeeze(data['positionalAnalysis']['displacement']['dy'])
    return t,x,y,speeds,dt,dx,dy


def gen_2d_histrogram(x, y, sigma, bins=80):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent, xedges, yedges


def get_heatmap_color_vectors(point_heatmap_value):
    # Convert the values into a actual color vectors
    cmap = cm.jet
    norm = Normalize(vmin=np.min(point_heatmap_value), vmax=np.max(point_heatmap_value))
    point_colors = cmap(norm(point_heatmap_value))
    return cmap, norm, point_colors

def bin_edges_to_midpoints(x):
    # Takes a set of N+1 edges and gets the N midpoints centered between each pair
    # See https://stackoverflow.com/questions/23855976/middle-point-of-each-pair-of-an-numpy-array
    return (x[1:] + x[:-1]) / 2


def process_chunk_equal_poritions_data_vectors(data_vectors_matrix, curr_view_window_length=30):
    # data_vectors_matrix: a N x M matrix where M is the number of equal length data vectors (of length N)
    num_data_vectors = np.shape(data_vectors_matrix)[0]
    data_vector_length = np.shape(data_vectors_matrix)[1]
     # Split the position data into equal sized chunks to be displayed at a single time. These will look like portions of the trajectory and be used to animate. # Chunk the data to create the animation.
    # curr_view_window_length = 150 # View 5 seconds at a time (30fps)
    # curr_view_window_length = 30 # View 5 seconds at a time (30fps)
    # The original length 324574 / 30 = 10819
    trimmed_elements = np.remainder(data_vector_length, curr_view_window_length) # Compute the number of elements that need to be droppped to be able to evently divide the original arrays into evenly sized chunks of length `curr_view_window_length`
    # e.g. np.remainder(324574, 150) = 124
    # drop 124 extra elements that make it no wrap evenly
    trimmed_length = data_vector_length - trimmed_elements
    # e.g. trimmed_length = 324574 - 124 # 324574 - 124
    other_reshaped_dimension = np.floor_divide(data_vector_length, curr_view_window_length) # e.g. 2163

    reshaped_data_matrix = np.zeros((num_data_vectors, other_reshaped_dimension, curr_view_window_length))
    for i in np.arange(num_data_vectors):
        reshaped_data_matrix[i,:,:] = data_vectors_matrix[i, 0:trimmed_length].reshape(other_reshaped_dimension, curr_view_window_length)

    return reshaped_data_matrix




def process_chunk_equal_poritions_data(t, x, y, speeds, dt, dx, dy, curr_view_window_length=30):
    # Split the position data into equal sized chunks to be displayed at a single time. These will look like portions of the trajectory and be used to animate. # Chunk the data to create the animation.
    # wraps process_chunk_equal_poritions_data_vectors(...) for a fixed input/output list.
    # data_vectors_matrix = np.vstack([t, x, y, speeds, dt, dx, dy]) # pack the variables of interest into the data_vector_matrix
    outputMatrix_fixedSegements = process_chunk_equal_poritions_data_vectors(np.vstack([t, x, y, speeds, dt, dx, dy]), curr_view_window_length)
    # print('shape - outputMatrix_fixedSegements: {}'.format(np.shape(outputMatrix_fixedSegements)))
    num_data_vectors = np.shape(outputMatrix_fixedSegements)[0]
    # unpack the result
    t_fixedSegements,x_fixedSegements,y_fixedSegements,speeds_fixedSegements,dt_fixedSegements,dx_fixedSegements,dy_fixedSegements = [np.squeeze(outputMatrix_fixedSegements[i,:,:]) for i in np.arange(num_data_vectors)]
    # print('shapes - t_fixedSegements: {}, x_fixedSegements: {}, y_fixedSegements: {}'.format(np.shape(t_fixedSegements), np.shape(x_fixedSegements), np.shape(y_fixedSegements)))
    return t_fixedSegements,x_fixedSegements,y_fixedSegements,speeds_fixedSegements,dt_fixedSegements,dx_fixedSegements,dy_fixedSegements

def extract_spike_timeseries(spike_cell):
    return spike_cell[:,1] # Extract only the first column that refers to the data.
    
def get_filtered_window(spike_list, spike_positions_list, min_timestep=0, max_timestep=400):
    num_cells = len(spike_list)
    active_spike_indices = [np.squeeze(np.where((min_timestep <= spike_timeseries) & (spike_timeseries <= max_timestep))) for spike_timeseries in spike_list]
    active_spike_list = [spike_list[i][active_spike_indices[i]] for i in np.arange(num_cells)]
    active_spike_positions_list = [spike_positions_list[i][:, active_spike_indices[i].T] for i in np.arange(num_cells)]
    return active_spike_indices, active_spike_list, active_spike_positions_list

