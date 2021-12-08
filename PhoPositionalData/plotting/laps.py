#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
from itertools import islice # for Pagination class
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def add_arrow(line, position=None, position_mode='rel', direction='right', size=15, color=None):
    """
    add an arrow to a Matplotlib line object, such as a line2D.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    position_mode: 
        # position_mode='rel': means a value between 0.0 (representing the start of the line) and 1.0 (the end) is provided.
        # position_mode='abs': means an absolute point is provided in [x, y] coordinates. Finds the nearest location to that point on the line.
        # position_mode='index': means an index into the line data is provided
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    
    Usage:
                add_arrow(line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                add_arrow(line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                add_arrow(line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
                add_arrow(line[0], position=curr_lap_endpoint, position_mode='abs', direction='right', size=50, color='blue')
                add_arrow(line[0], position=None, position_mode='rel', direction='right', size=50, color='blue')
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()
    num_points = len(xdata)
    
    if position is None:
        position = xdata.mean()
        position_mode = 'abs_nearest'
        
    if position_mode == 'rel':
        # position_mode='rel': means a value between 0.0 (representing the start of the line) and 1.0 (the end) is provided.
        
        start_ind = np.round(float(position) * float(num_points))
    elif position_mode == 'abs':
        # position_mode='abs': means an absolute point is provided in [x, y] coordinates. Finds the nearest location to that point on the line.
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position[0]))
       
    elif position_mode == 'abs_nearest':
        # position_mode='abs': means an absolute point is provided in [x, y] coordinates. Finds the nearest location to that point on the line.
        # find closest index
        start_ind = np.argmin(np.absolute(xdata - position))
        
    elif position_mode == 'index':
        # position_mode='index': means an index into the line data is provided
        start_ind = position        
    else:
        raise ValueError
        
    # compute the end index based on the arrow direction:
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    # fix out of bound indicies:
    if (end_ind < 0):
        end_ind = 0
        start_ind = 1
    elif (end_ind >= num_points):
        end_ind = num_points - 1
        start_ind = num_points - 2
    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


def plot_lap_trajectories_2d(sess, curr_num_subplots=5, active_page_index=0):
    """ Plots a MatplotLib 2D Figure with each lap being shown in one of its subplots """
    def _chunks(iterable, size=10):
        iterator = iter(iterable)
        for first in iterator:    # stops when iterator is depleted
            def chunk():          # construct generator for next chunk
                yield first       # yield element from for loop
                for more in islice(iterator, size - 1):
                    yield more    # yield more elements from the iterator
            yield chunk()         # in outer generator, yield next chunk

    def _compute_laps_position_data(sess):
        curr_position_df = sess.compute_position_laps()
        lap_specific_position_dfs = [curr_position_df.groupby('lap').get_group(i)[['t','x','y','lin_pos']] for i in sess.laps.lap_id] # dataframes split for each ID:
        return curr_position_df, lap_specific_position_dfs
        
    def _build_laps_multiplotter(nfields, linear_plot_data=None):
        linear_plotter_indicies = np.arange(nfields)
        fixed_columns = 2
        needed_rows = int(np.ceil(nfields / fixed_columns))
        row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        mp, axs = plt.subplots(needed_rows, fixed_columns, sharex=True, sharey=True) #ndarray (5,2)
        mp.set_size_inches(18.5, 26.5)
        for a_linear_index in linear_plotter_indicies:
            curr_row = row_column_indicies[0][a_linear_index]
            curr_col = row_column_indicies[1][a_linear_index]
            axs[curr_row][curr_col].plot(linear_plot_data[a_linear_index][0,:], linear_plot_data[a_linear_index][1,:], c='k', alpha=0.2)
            
        return mp, axs, linear_plotter_indicies, row_column_indicies
    
    def _add_specific_lap_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, laps_position_traces, lap_time_ranges, use_time_gradient_line=True):
        # Add the lap trajectory:                            
        for a_linear_index in linear_plotter_indicies:
            curr_lap_id = active_page_laps_ids[a_linear_index]
            curr_row = row_column_indicies[0][a_linear_index]
            curr_col = row_column_indicies[1][a_linear_index]
            curr_lap_time_range = lap_time_ranges[curr_lap_id]
            curr_lap_label_text = 'Lap[{}]: t({:.2f}, {:.2f})'.format(curr_lap_id, curr_lap_time_range[0], curr_lap_time_range[1])
            curr_lap_num_points = len(laps_position_traces[curr_lap_id][0,:])
            if use_time_gradient_line:
                # Create a continuous norm to map from data points to colors
                curr_lap_timeseries = np.linspace(curr_lap_time_range[0], curr_lap_time_range[-1], len(laps_position_traces[curr_lap_id][0,:]))
                norm = plt.Normalize(curr_lap_timeseries.min(), curr_lap_timeseries.max())
                # needs to be (numlines) x (points per line) x 2 (for x and y)
                points = np.array([laps_position_traces[curr_lap_id][0,:], laps_position_traces[curr_lap_id][1,:]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                # Set the values used for colormapping
                lc.set_array(curr_lap_timeseries)
                lc.set_linewidth(2)
                lc.set_alpha(0.85)
                line = axs[curr_row][curr_col].add_collection(lc)
                # add_arrow(line)
            else:
                line = axs[curr_row][curr_col].plot(laps_position_traces[curr_lap_id][0,:], laps_position_traces[curr_lap_id][1,:], c='k', alpha=0.85)
                # curr_lap_endpoint = curr_lap_position_traces[curr_lap_id][:,-1].T
                add_arrow(line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                add_arrow(line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                add_arrow(line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
                # add_arrow(line[0], position=curr_lap_endpoint, position_mode='abs', direction='right', size=50, color='blue')
                # add_arrow(line[0], position=None, position_mode='rel', direction='right', size=50, color='blue')
            # add lap text label
            axs[curr_row][curr_col].text(250, 126, curr_lap_label_text, horizontalalignment='right', size=12)
            # PhoWidgetHelper.perform_add_text(p[curr_row, curr_col], curr_lap_label_text, name='lblLapIdIndicator')

    # Compute required data from session:
    curr_position_df, lap_specific_position_dfs = _compute_laps_position_data(sess)
    laps_position_traces_list = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
    laps_time_range_list = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in lap_specific_position_dfs]
    
    num_laps = len(sess.laps.lap_id)
    linear_lap_index = np.arange(num_laps)
    lap_time_ranges = dict(zip(sess.laps.lap_id, laps_time_range_list))
    lap_position_traces = dict(zip(sess.laps.lap_id, laps_position_traces_list))
    
    all_maze_positions = curr_position_df[['x','y']].to_numpy().T # (2, 59308)
    # np.shape(all_maze_positions)
    all_maze_data = [all_maze_positions for i in  np.arange(curr_num_subplots)] # repeat the maze data for each subplot. (2, 593080)
    p, axs, linear_plotter_indicies, row_column_indicies = _build_laps_multiplotter(curr_num_subplots, all_maze_data)
    # generate the pages
    laps_pages = [list(chunk) for chunk in _chunks(sess.laps.lap_id, curr_num_subplots)]
    active_page_laps_ids = laps_pages[active_page_index]
    _add_specific_lap_trajectory(p, axs, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, lap_position_traces, lap_time_ranges, use_time_gradient_line=True)
    plt.ylim((125, 152))
    return p, axs, laps_pages