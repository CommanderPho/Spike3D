# LapsVisualizationMixin.py
import numpy as np
import pandas as pd
import pyvista as pv

class LapsVisualizationMixin:

    @staticmethod
    def lines_from_points(points):
		"""Given an array of points, make a line set"""
		poly = pv.PolyData()
		poly.points = points
		cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
		cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
		cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
		poly.lines = cells
		return poly


	@staticmethod
	def get_lap_times(session, curr_lap_id):
		return session.laps.get_lap_times(curr_lap_id)

	@staticmethod
	def get_lap_position(session, curr_lap_id):
		curr_position_df = session.position.to_dataframe()
		curr_lap_t_start, curr_lap_t_stop = get_lap_times(curr_lap_id)
		print('lap[{}]: ({}, {}): '.format(curr_lap_id, curr_lap_t_start, curr_lap_t_stop))

		curr_lap_position_df_is_included = curr_position_df['t'].between(curr_lap_t_start, curr_lap_t_stop, inclusive=True) # returns a boolean array indicating inclusion in teh current lap
		curr_lap_position_df = curr_position_df[curr_lap_position_df_is_included] 
		# curr_position_df.query('-0.5 <= t < 0.5')
		curr_lap_position_traces = curr_lap_position_df[['x','y']].to_numpy().T
		print('\t {} positions.'.format(np.shape(curr_lap_position_traces)))
		# print('\t {} spikes.'.format(curr_lap_num_spikes))
		return curr_lap_position_traces


	@staticmethod
	def plot_lap_trajectory_path(interactiveDataExplorer, curr_lap_position_traces):
		num_lap_samples = np.shape(curr_lap_position_traces)[1]
		lap_fixed_z = np.full_like(curr_lap_position_traces[0,:], 0.9)
		plot_name = 'lap_location_trail'
		fade_values = interactiveDataExplorer.params.active_trail_opacity_values
		# size_values = ipspikesDataExplorer.params.active_trail_size_values
		trail_fade_values = None
		size_values = None
		trail_fade_values = np.linspace(0.0, 0.6, num_lap_samples)
		size_values = np.linspace(0.2, 0.6, num_lap_samples) # fade from a scale of 0.2 to 0.6
		interactiveDataExplorer.perform_plot_location_trail(plot_name, curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], lap_fixed_z,
													trail_fade_values=trail_fade_values, trail_point_size_values=size_values,
													render=True, color='red')


	
	@staticmethod
	def plot_lap_trajectory_path_spline(interactiveDataExplorer, curr_lap_position_traces):
		num_lap_samples = np.shape(curr_lap_position_traces)[1]
		lap_fixed_z = np.full_like(curr_lap_position_traces[0,:], 0.9)
		curr_lap_points = np.column_stack((curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], lap_fixed_z))
		plot_name = 'lap_location_trail_spline'
		fade_values = interactiveDataExplorer.params.active_trail_opacity_values
		# size_values = ipspikesDataExplorer.params.active_trail_size_values
		trail_fade_values = None
		size_values = None
		trail_fade_values = np.linspace(0.0, 0.6, num_lap_samples)
		size_values = np.linspace(0.2, 0.6, num_lap_samples) # fade from a scale of 0.2 to 0.6
		line = LapsVisualizationMixin.lines_from_points(curr_lap_points)
		line["scalars"] = np.arange(line.n_points)
		tube = line.tube(radius=0.5)
		# tube.plot(smooth_shading=True)
		interactiveDataExplorer.p.add_mesh(tube, name=plot_name, render_lines_as_tubes=True, show_scalar_bar=False, color='red')
  
  
	@staticmethod
	def plot_lap_trajectory_path_spline(interactiveDataExplorer, curr_lap_position_traces, curr_lap_id):
		num_lap_samples = np.shape(curr_lap_position_traces)[1]
		lap_fixed_z = np.full_like(curr_lap_position_traces[0,:], 0.9 + (0.45 * curr_lap_id))
		curr_lap_points = np.column_stack((curr_lap_position_traces[0,:], curr_lap_position_traces[1,:], lap_fixed_z))
		plot_name = 'lap_location_trail_spline[{}]'.format(int(curr_lap_id))
		trail_fade_values = np.linspace(0.0, 0.6, num_lap_samples)
		size_values = np.linspace(0.2, 0.6, num_lap_samples) # fade from a scale of 0.2 to 0.6
		line = LapsVisualizationMixin.lines_from_points(curr_lap_points)
		line["scalars"] = np.arange(line.n_points)
		tube = line.tube(radius=0.2)
		# tube.plot(smooth_shading=True)
		interactiveDataExplorer.p.add_mesh(tube, name=plot_name, render_lines_as_tubes=False, show_scalar_bar=False, cmap='fire', lighting=False, render=False)