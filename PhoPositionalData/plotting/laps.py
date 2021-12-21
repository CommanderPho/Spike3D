#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
from itertools import islice # for Pagination class
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, BrokenBarHCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data, perform_plot_flat_arena
from PhoGui.InteractivePlotter.LapsVisualizationMixin import LapsVisualizationMixin
from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper
import pyvista as pv
import pyvistaqt as pvqt

""" 
from PhoPositionalData.plotting.laps import plot_lap_trajectories_2d
# Complete Version:
fig, axs, laps_pages = plot_lap_trajectories_2d(sess, curr_num_subplots=len(sess.laps.lap_id), active_page_index=0)
# Paginated Version:
fig, axs, laps_pages = plot_lap_trajectories_2d(sess, curr_num_subplots=22, active_page_index=0)
fig, axs, laps_pages = plot_lap_trajectories_2d(sess, curr_num_subplots=22, active_page_index=1)



"""
def _plot_helper_add_arrow(line, position=None, position_mode='rel', direction='right', size=15, color=None):
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

def _plot_helper_add_span_where_ranges(pos_t: np.ndarray, pos_where_even_lap_indicies, pos_where_odd_lap_indicies, curr_ax: plt.axes):
    """ Span_where implementation: Draws colored spans indicating the lap that is active during a given time interval. 
    
    Usage Example:
        pos_df_is_nonNaN_lap = np.logical_not(np.isnan(pos_df.lap))
        pos_df_is_even_lap = np.logical_and(pos_df_is_nonNaN_lap, (np.remainder(pos_df.lap, 2) == 0))
        pos_df_is_odd_lap = np.logical_and(pos_df_is_nonNaN_lap, (np.remainder(pos_df.lap, 2) != 0))
    """
    curr_span_ymin = curr_ax.get_ylim()[0]
    curr_span_ymax = curr_ax.get_ylim()[1]
    collection = BrokenBarHCollection.span_where(
        pos_t, ymin=curr_span_ymin, ymax=curr_span_ymax, where=pos_where_even_lap_indicies, facecolor='green', alpha=0.25)
    curr_ax.add_collection(collection)
    collection = BrokenBarHCollection.span_where(
        pos_t, ymin=curr_span_ymin, ymax=curr_span_ymax, where=pos_where_odd_lap_indicies, facecolor='red', alpha=0.25)
    curr_ax.add_collection(collection)
    
def _build_included_mask(mask_shape, crossing_beginings, crossing_endings):
    # included_mask = np.full_like(pos_df['x'], False)
    included_mask = np.full(mask_shape, False) 
    num_items = len(crossing_beginings)
    included_index_ranges = [np.arange(crossing_beginings[i], crossing_endings[i]) for i in np.arange(num_items)]
    for aRange in included_index_ranges:
        included_mask[aRange] = True
    return included_mask, included_index_ranges


def _plot_helper_render_laps(pos_t_rel_seconds, pos_value, crossing_beginings, crossing_midpoints, crossing_endings, color='g', ax=None):
    """ renders a set of estimated laps with the provided settings
    Usage:
        fig, out_axes_list = plot_position_curves_figure(position_obj, include_velocity=False, include_accel=False)
        _plot_helper_render_lap(pos_df['t'].to_numpy(), pos_df['x'].to_numpy(), desc_crossing_beginings, desc_crossing_midpoints, desc_crossing_endings, color='r', ax=out_axes_list[0])
        _plot_helper_render_lap(pos_df['t'].to_numpy(), pos_df['x'].to_numpy(), asc_crossing_beginings, asc_crossing_midpoints, asc_crossing_endings, color='g', ax=out_axes_list[0])
    """
    if ax is None:
        ax = plt.gca()
    # Plots the computed midpoint center-crossing for each lap. This is the basis of the calculation initially.
    ax.scatter(pos_t_rel_seconds[crossing_midpoints], pos_value[crossing_midpoints], s=15, c=color)    
    # Plots the concrete vertical lines denoting the start/end of each lap
    ax.vlines(pos_t_rel_seconds[crossing_beginings], 0, 1, transform=ax.get_xaxis_transform(), colors=color)
    ax.vlines(pos_t_rel_seconds[crossing_endings], 0, 1, transform=ax.get_xaxis_transform(), colors=color)
    # Plot the ranges for the ascending and descending laps:
    curr_included_mask, curr_included_index_ranges = _build_included_mask(np.shape(pos_value), crossing_beginings, crossing_endings)
    collection = BrokenBarHCollection.span_where(pos_t_rel_seconds, ymin=0, ymax=1, transform=ax.get_xaxis_transform(), where=curr_included_mask, facecolor=color, alpha=0.35)
    ax.add_collection(collection)

def plot_position_curves_figure(position_obj, include_velocity=True, include_accel=False, figsize=(24, 10)):
    """ Renders a figure with a position curve and optionally its higher-order derivatives """
    num_subplots = 1
    out_axes_list = []
    if include_velocity:
        num_subplots = num_subplots + 1
    if include_accel:
        num_subplots = num_subplots + 1
    subplots=(num_subplots, 1)
    fig = plt.figure(figsize=figsize, clear=True)
    gs = plt.GridSpec(subplots[0], subplots[1], figure=fig, hspace=0.02)
    
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(position_obj.time, position_obj.x, 'k')
    ax0.set_ylabel('pos_x')
    out_axes_list.append(ax0)
    
    if include_velocity:
        ax1 = fig.add_subplot(gs[1])
        # ax1.plot(position_obj.time, pos_df['velocity_x'], 'grey')
        # ax1.plot(position_obj.time, pos_df['velocity_x_smooth'], 'r')
        ax1.plot(position_obj.time, position_obj._data['velocity_x_smooth'], 'k')
        ax1.set_ylabel('Velocity_x')
        ax0.set_xticklabels([]) # this is intensionally ax[i-1], as we want to disable the tick labels on above plots        
        out_axes_list.append(ax1)

    if include_accel:  
        ax2 = fig.add_subplot(gs[2])
        # ax2.plot(position_obj.time, position_obj.velocity)
        # ax2.plot(position_obj.time, pos_df['velocity_x'])
        ax2.plot(position_obj.time, position_obj._data['acceleration_x'], 'k')
        # ax2.plot(position_obj.time, pos_df['velocity_y'])
        ax2.set_ylabel('Higher Order Terms')
        ax1.set_xticklabels([]) # this is intensionally ax[i-1], as we want to disable the tick labels on above plots
        out_axes_list.append(ax2)
    
    # Shared:
    # ax0.get_shared_x_axes().join(ax0, ax1)
    ax0.get_shared_x_axes().join(*out_axes_list)
    ax0.set_xticklabels([])
    ax0.set_xlim([position_obj.time[0], position_obj.time[-1]])

    return fig, out_axes_list

    

    
    
def plot_laps_2d(sess):
    pos_df = sess.compute_position_laps() # ensures the laps are computed if they need to be:
    position_obj = sess.position
    position_obj.compute_higher_order_derivatives()
    pos_df = position_obj.to_dataframe()
    
    
    ## non-pre-filtered version, also doesn't create a duplicate dataframe:
    pos_df_is_nonNaN_lap = np.logical_not(np.isnan(pos_df.lap))
    pos_df_is_even_lap = np.logical_and(pos_df_is_nonNaN_lap, (np.remainder(pos_df.lap, 2) == 0))
    pos_df_is_odd_lap = np.logical_and(pos_df_is_nonNaN_lap, (np.remainder(pos_df.lap, 2) != 0))
    
    figsize=(24, 10)
    subplots=(3, 1)
    fig = plt.figure(figsize=figsize, clear=True)
    gs = plt.GridSpec(subplots[0], subplots[1], figure=fig, hspace=0.02)
    # fig.subplots_adjust(hspace=0.4)
    # position_obj = active_epoch_session.position

    ax0 = fig.add_subplot(gs[0])
    # ax0.plot(position_obj.time, position_obj.linear_pos)
    # ax0.set_ylabel('linear_pos')
    ax0.plot(position_obj.time, position_obj.x, 'k')
    ax0.set_ylabel('pos_x')


    ax1 = fig.add_subplot(gs[1])
    ax1.plot(position_obj.time, pos_df['velocity_x'], 'k')
    ax1.set_ylabel('Velocity_x')
    
    ax2 = fig.add_subplot(gs[2])
    # ax2.plot(position_obj.time, position_obj.velocity)
    # ax2.plot(position_obj.time, pos_df['velocity_x'])
    ax2.plot(position_obj.time, pos_df['acceleration_x'], 'k')
    # ax2.plot(position_obj.time, pos_df['velocity_y'])
    ax2.set_ylabel('Higher Order Terms')

    # Shared:
    ax0.get_shared_x_axes().join(ax0, ax1, ax2)
    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax0.set_xlim([position_obj.time[0], position_obj.time[-1]])
    ## end plot.
    
    ## Draw on top of the existing position curves with the lap colors:
    curr_even_lap_dir_points = pos_df[pos_df_is_even_lap][['t','x']].to_numpy()
    ax0.scatter(curr_even_lap_dir_points[:,0], curr_even_lap_dir_points[:,1], s=0.5, c='g')
    curr_odd_lap_dir_points = pos_df[pos_df_is_odd_lap][['t','x']].to_numpy()
    ax0.scatter(curr_odd_lap_dir_points[:,0], curr_odd_lap_dir_points[:,1], s=0.5, c='r')

    ## Draw the horizontal spans for each subplot:
    _plot_helper_add_span_where_ranges(pos_df.t.to_numpy(), pos_df_is_even_lap, pos_df_is_odd_lap, ax0)
    _plot_helper_add_span_where_ranges(pos_df.t.to_numpy(), pos_df_is_even_lap, pos_df_is_odd_lap, ax1)
    _plot_helper_add_span_where_ranges(pos_df.t.to_numpy(), pos_df_is_even_lap, pos_df_is_odd_lap, ax2)
    ax0.set_title('Laps')
    # fig.suptitle('Laps', fontsize=22)



def plot_lap_trajectories_3d(sess, curr_num_subplots=5, active_page_index=0):
    """ Plots a PyVista Qt Multiplotter with several overhead 3D views, each showing a specific lap over the maze in one of its subplots 
    Usage: 
        p, laps_pages = plot_lap_trajectories_3d(sess, curr_num_subplots=10, active_page_index=1)
        p.show()
    """
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
        fixed_columns = 5
        needed_rows = int(np.ceil(nfields / fixed_columns))
        row_column_indicies = np.unravel_index(linear_plotter_indicies, (needed_rows, fixed_columns)) # inverse is: np.ravel_multi_index(row_column_indicies, (needed_rows, fixed_columns))
        mp = pvqt.MultiPlotter(nrows=needed_rows, ncols=fixed_columns, show=False, title='Laps Muliplotter', toolbar=False, menu_bar=False, editor=False)
        # print('linear_plotter_indicies: {}\n row_column_indicies: {}\n'.format(linear_plotter_indicies, row_column_indicies))
        for a_linear_index in linear_plotter_indicies:
            # print('a_linear_index: {}, row_column_indicies[0][a_linear_index]: {}, row_column_indicies[1][a_linear_index]: {}'.format(a_linear_index, row_column_indicies[0][a_linear_index], row_column_indicies[1][a_linear_index]))
            curr_row = row_column_indicies[0][a_linear_index]
            curr_col = row_column_indicies[1][a_linear_index]
            if linear_plot_data is None:
                mp[curr_row, curr_col].add_mesh(pv.Sphere())
            else:
                mp[curr_row, curr_col].add_mesh(linear_plot_data[a_linear_index], name='maze_bg', color="black", render=False)
        return mp, linear_plotter_indicies, row_column_indicies

    # Plot the flat arena
    def __build_flat_map_plot_data(x, y):
        # Builds the flat base maze map that the other data will be plot on top of
        z = np.zeros_like(x)
        point_cloud = np.vstack((x, y, z)).T
        pdata = pv.PolyData(point_cloud)
        pdata['occupancy heatmap'] = np.arange(np.shape(point_cloud)[0])
        geo = pv.Circle(radius=0.5)
        pc = pdata.glyph(scale=False, geom=geo)
        return pdata, pc
    
    def _add_specific_lap_trajectory(p, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, curr_lap_position_traces, curr_lap_time_range):
        # Add the lap trajectory:
        for a_linear_index in linear_plotter_indicies:
            curr_lap_id = active_page_laps_ids[a_linear_index]
            curr_row = row_column_indicies[0][a_linear_index]
            curr_col = row_column_indicies[1][a_linear_index]
            LapsVisualizationMixin.plot_lap_trajectory_path_spline(p[curr_row, curr_col], curr_lap_position_traces[curr_lap_id], a_linear_index)
            curr_lap_label_text = 'Lap[{}]: t({:.2f}, {:.2f})'.format(curr_lap_id, curr_lap_time_range[curr_lap_id][0], curr_lap_time_range[curr_lap_id][1]) 
            PhoWidgetHelper.perform_add_text(p[curr_row, curr_col], curr_lap_label_text, name='lblLapIdIndicator')

    # Compute required data from session:
    curr_position_df, lap_specific_position_dfs = _compute_laps_position_data(sess)
    curr_lap_position_traces = [lap_pos_df[['x','y']].to_numpy().T for lap_pos_df in lap_specific_position_dfs]
    curr_lap_time_range = [[lap_pos_df[['t']].to_numpy()[0].item(), lap_pos_df[['t']].to_numpy()[-1].item()] for lap_pos_df in lap_specific_position_dfs]

    all_maze_positions = curr_position_df[['x','y']].to_numpy().T # (2, 59308)
    # np.shape(all_maze_positions)
    pdata_maze_shared, pc_maze_shared = __build_flat_map_plot_data(all_maze_positions[0,:], all_maze_positions[1,:])
    all_maze_data = np.full((curr_num_subplots,), pc_maze_shared) # repeat the maze data for each subplot
    p, linear_plotter_indicies, row_column_indicies = _build_laps_multiplotter(curr_num_subplots, all_maze_data)
    # generate the pages
    laps_pages = [list(chunk) for chunk in _chunks(sess.laps.lap_id, curr_num_subplots)]
    active_page_laps_ids = laps_pages[active_page_index]
    _add_specific_lap_trajectory(p, linear_plotter_indicies, row_column_indicies, active_page_laps_ids, curr_lap_position_traces, curr_lap_time_range)
    return p, laps_pages



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
                _plot_helper_add_arrow(line[0], position=0, position_mode='index', direction='right', size=20, color='green') # start
                _plot_helper_add_arrow(line[0], position=None, position_mode='index', direction='right', size=20, color='yellow') # middle
                _plot_helper_add_arrow(line[0], position=curr_lap_num_points, position_mode='index', direction='right', size=20, color='red') # end
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