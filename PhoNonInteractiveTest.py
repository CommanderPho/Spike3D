# PhoNonInteractiveTest.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
# %load_ext autoreload
# %autoreload 2
import sys
import importlib
import numpy as np
import pandas as pd

#interactive plotting in separate window
# %matplotlib qt
# %matplotlib inline
# %matplotlib notebook
# %matplotlib widget
# %matplotlib qt

# %matplotlib widget
from matplotlib.colors import ListedColormap
from copy import deepcopy
# from PyQt5 import QtWidgets, uic
# from pyqt6 import QApplication
from IPython.core.display import display, HTML
display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))
from IPython.lib.pretty import pretty, pprint
# from pretty import pprint

# Pandas display options
# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199
pd.set_option('display.width', 1000)


from neuropy import core
from neuropy.core.session.data_session_loader import DataSessionLoader
from neuropy.core.session.dataSession import DataSession
from neuropy.core.epoch import Epoch
from neuropy.core.epoch import NamedTimerange
from neuropy.core import Laps
from neuropy.core import Position
from neuropy.core import FlattenedSpiketrains
from neuropy.core import Neurons
from neuropy.utils.misc import print_seconds_human_readable
from neuropy.plotting import plot_raster
from neuropy.analyses.placefields import PlacefieldComputationParameters, PfND, compute_placefields_masked_by_epochs, compute_placefields_as_needed
from neuropy.utils.debug_helpers import debug_print_placefield, debug_print_spike_counts, debug_print_subsession_neuron_differences

from pyphoplacecellanalysis.PhoPositionalData.load_exported import *
# from pyphoplacecellanalysis.PhoPositionalData.process_data import process_positionalAnalysis_data, gen_2d_histrogram, get_heatmap_color_vectors, process_chunk_equal_poritions_data, extract_spike_timeseries
from pyphoplacecellanalysis.PhoPositionalData.process_data import *
from pyphoplacecellanalysis.PhoPositionalData.plot_data import *
from pyphoplacecellanalysis.PhoPositionalData.plotting.animations import * # make_mp4_from_plotter
from pyphoplacecellanalysis.PhoPositionalData.import_data import * # build_spike_positions_list, build_cellID_reverse_lookup_map

from PendingNotebookCode import build_configs, build_units_colormap, build_placefield_multiplotter, process_by_good_placefields, estimation_session_laps, partition

# from pyphoplacecellanalysis.PhoPositionalData.debug_helpers import debug_print_placefield, debug_print_spike_counts



""" For running in IPython:
	
    from PhoNonInteractiveTest import PhoNonInteractiveTest
    from neuropy.plotting.ratemaps import plot_ratemap_2D, enumTuningMap2DPlotVariables, enumTuningMap2DPlotMode

	test = PhoNonInteractiveTest()
	active_epoch_placefields1D, active_epoch_placefields2D, even_lap_specific_placefields1D, even_lap_specific_placefields2D, odd_lap_specific_placefields1D, odd_lap_specific_placefields2D, any_lap_specific_placefields1D, any_lap_specific_placefields2D, active_config, active_epoch_session, good_placefields_session = PhoNonInteractiveTest.run(test.sess, test.active_sess_config)
	

"""

class PhoNonInteractiveTest:
	def __init__(self, basedir = r'W:\data\KDIBA\gor01\one\2006-6-07_11-26-53') -> None:
		# KDiba Old Format:
		## Data must be pre-processed using the MATLAB script located here: 
		# R:\data\KDIBA\gor01\one\IIDataMat_Export_ToPython_2021_11_23.m
		# From pre-computed .mat files:
		if basedir is None:
			# 07: 
			self.basedir = r'W:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
			# # ## 08:
			# basedir = r'W:\data\KDIBA\gor01\one\2006-6-08_14-26-15'
		print(f'loading basedir {basedir}...')
		self.load(self.basedir)
		print('\t session dataframe spikes: {}\nsession.neurons.n_spikes summed: {}\n'.format(self.sess.spikes_df.shape, np.sum(self.sess.neurons.n_spikes)))
		## Estimate the Session's Laps data using my algorithm from the loaded position data.
		self.sess = estimation_session_laps(self.sess)

	def load(self, basedir):
		self.sess = DataSessionLoader.kdiba_old_format_session(basedir)
		self.active_sess_config = self.sess.config
		self.session_name = self.sess.name
 
	@classmethod
	def run(cls, sess, active_sess_config):
		""" 
		Usage:
		 	 active_epoch_placefields1D, active_epoch_placefields2D, even_lap_specific_placefields1D, even_lap_specific_placefields2D, odd_lap_specific_placefields1D, odd_lap_specific_placefields2D, any_lap_specific_placefields1D, any_lap_specific_placefields2D, active_config, active_epoch_session, good_placefields_session = PhoNonInteractiveTest.run(test.sess, test.active_sess_config)
     	"""
		lap_specific_epochs = sess.laps.as_epoch_obj()
		any_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(len(sess.laps.lap_id))])
		even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(sess.laps.lap_id), 2)])
		odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(sess.laps.lap_id), 2)])

		sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
		# sess.epochs.to_dataframe()
		# active_epoch = sess.epochs.get_named_timerange('maze1')
		# print('active_epoch: {}'.format(active_epoch))
		# active_epoch = sess.epochs.get_named_timerange('maze2')
		active_epoch = NamedTimerange(name='maze', start_end_times=[sess.epochs['maze1'][0], sess.epochs['maze2'][1]])
		active_subplots_shape = (1,1) # Single subplot
		# active_subplots_shape = '1|2' # 1 subplot on left, two on right
		active_config = build_configs(active_sess_config, active_epoch, active_subplots_shape = active_subplots_shape)

		## All Spikes:
		active_epoch_session = sess.filtered_by_neuron_type('pyramidal').filtered_by_epoch(active_epoch)
		print_subsession_neuron_differences(sess.neurons, active_epoch_session.neurons)
		# print(sess.neurons.n_spikes)

		# # ## Lap_specific Spikes Only:
		# active_lap_specific_epoch_session = lap_specific_session.filtered_by_neuron_type('pyramidal').filtered_by_epoch(active_epoch)
		# print_subsession_neuron_differences(lap_specific_session.neurons, active_lap_specific_epoch_session.neurons)
		# # print(active_lap_specific_epoch_session.neurons.n_spikes)

		## Configure Placefield Calc:
		should_display_2D_plots = False

		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=2, smooth=0.5, frate_thresh=2.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=10, smooth=0.5, frate_thresh=2.0) # works well
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=2.5, smooth=1.5, frate_thresh=2.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(0.5, 0.5), frate_thresh=0.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0.0, grid_bin=(3, 4), smooth=(2, 1), frate_thresh=2.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 10), smooth=(0.5, 0.5), frate_thresh=2.0) ## Works well for 2D Placemaps
		# height: 20.0
		# width: 250.0
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0, grid_bin=(2.0, 0.2), smooth=(0.5, 0.5), frate_thresh=2.0) ## Extremely Slow
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0, grid_bin=(2.0, 1.0), smooth=(0.5, 0.5), frate_thresh=2.0) ## Very slow, doesn't work

		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(0.0, 0.0), frate_thresh=2.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(0.1, 0.1), frate_thresh=2.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(1.0, 10.0), frate_thresh=2.0)

		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0.0, grid_bin=(25, 9), smooth=(0.0, 0.0), frate_thresh=2.0)
		# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0.0, grid_bin=(5, 3), smooth=(0.0, 0.0), frate_thresh=2.0)
		active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0.0, grid_bin=(5, 3), smooth=(0.0, 0.0), frate_thresh=0.1) # TODO: FIXME: BUG: when frate_thresh=0.0, there are 0 good placefield_neuronIDs for all computations!


		active_epoch_placefields1D, active_epoch_placefields2D = compute_placefields_masked_by_epochs(active_epoch_session, active_config, included_epochs=None, should_display_2D_plots=should_display_2D_plots)
		even_lap_specific_placefields1D, even_lap_specific_placefields2D = compute_placefields_masked_by_epochs(active_epoch_session, active_config, included_epochs=even_lap_specific_epochs, should_display_2D_plots=should_display_2D_plots)
		odd_lap_specific_placefields1D, odd_lap_specific_placefields2D = compute_placefields_masked_by_epochs(active_epoch_session, active_config, included_epochs=odd_lap_specific_epochs, should_display_2D_plots=should_display_2D_plots)
		any_lap_specific_placefields1D, any_lap_specific_placefields2D = compute_placefields_masked_by_epochs(active_epoch_session, active_config, included_epochs=any_lap_specific_epochs, should_display_2D_plots=should_display_2D_plots)
		# Compare the results
		# debug_print_ratemap(active_epoch_placefields1D.ratemap)
		# num_spikes_per_spiketrain = np.array([np.shape(a_spk_train)[0] for a_spk_train in active_epoch_placefields1D.spk_t])
		# num_spikes_per_spiketrain
		# print('placefield_neuronID_spikes: {}; ({} total spikes)'.format(num_spikes_per_spiketrain, np.sum(num_spikes_per_spiketrain)))

		debug_print_placefield(active_epoch_placefields1D)
		debug_print_placefield(any_lap_specific_placefields1D)
		debug_print_placefield(even_lap_specific_placefields1D)
		debug_print_placefield(odd_lap_specific_placefields1D)

		# Get the cell IDs that have a good place field mapping:
		active_placefields = deepcopy(any_lap_specific_placefields2D) # not changed this from the default placefields2D object
		good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
		good_placefield_tuple_neuronIDs = active_placefields.neuron_extended_ids

		## Filter by neurons with good placefields only:
		good_placefields_session = active_epoch_session.get_by_id(good_placefield_neuronIDs) # Filter by good placefields only, and this fetch also ensures they're returned in the order of sorted ascending index ([ 2  3  5  7  9 12 18 21 22 23 26 27 29 34 38 45 48 53 57])
			
		pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
		active_config.plotting_config.pf_sort_ind = pf_sort_ind
		active_config.plotting_config.pf_colors = pf_colors
		active_config.plotting_config.active_cells_colormap = pf_colormap
		active_config.plotting_config.active_cells_listed_colormap = ListedColormap(active_config.plotting_config.active_cells_colormap)
  
		return active_epoch_placefields1D, active_epoch_placefields2D, even_lap_specific_placefields1D, even_lap_specific_placefields2D, odd_lap_specific_placefields1D, odd_lap_specific_placefields2D, any_lap_specific_placefields1D, any_lap_specific_placefields2D, active_config, active_epoch_session, good_placefields_session
  

