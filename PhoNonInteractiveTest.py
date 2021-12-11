# PhoNonInteractiveTest.py



class PhoNonInteractiveTest:
	def __init__(self) -> None:
		self.
    
    def load():
		# KDiba Old Format:
		## Data must be pre-processed using the MATLAB script located here: 
		# R:\data\KDIBA\gor01\one\IIDataMat_Export_ToPython_2021_11_23.m
		# From pre-computed .mat files:
		## 07: 
		# basedir = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
		# ## 08:
		basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'
		sess = DataSessionLoader.kdiba_old_format_session(basedir)
		active_sess_config = sess.config
		session_name = sess.name
 
	## Lap Only:
	lap_specific_epochs = sess.laps.as_epoch_obj()
	even_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(0, len(sess.laps.lap_id), 2)])
	odd_lap_specific_epochs = lap_specific_epochs.label_slice(lap_specific_epochs.labels[np.arange(1, len(sess.laps.lap_id), 2)])

	# Filter Session by Epoch:
 	sess.epochs.t_start = 22.26 # exclude the first short period where the animal isn't on the maze yet
	# sess.epochs.to_dataframe()
	active_epoch = sess.epochs.get_named_timerange('maze1')
	# print('active_epoch: {}'.format(active_epoch))
	# active_epoch = sess.epochs.get_named_timerange('maze2')
	# active_epoch = NamedTimerange(name='maze', start_end_times=[sess.epochs['maze1'][0], sess.epochs['maze2'][1]])
	active_subplots_shape = (1,1) # Single subplot
	# active_subplots_shape = '1|2' # 1 subplot on left, two on right                                                   
	active_config = build_configs(active_sess_config, active_epoch, active_subplots_shape = active_subplots_shape)

	## All Spikes:
	active_epoch_session = sess.filtered_by_neuron_type('pyramidal').filtered_by_epoch(active_epoch)
	print_subsession_neuron_differences(sess.neurons, active_epoch_session.neurons)
	# print(sess.neurons.n_spikes)

	# ## Lap_specific Spikes Only:
	# active_lap_specific_epoch_session = lap_specific_session.filtered_by_neuron_type('pyramidal').filtered_by_epoch(active_epoch)
	# print_subsession_neuron_differences(lap_specific_session.neurons, active_lap_specific_epoch_session.neurons)
	# print(active_lap_specific_epoch_session.neurons.n_spikes)
 
	# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=2, smooth=0.5)
	# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=10, smooth=0.5) # works well
	# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=2.5, smooth=1.5)
	# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 3), smooth=(0.5, 0.5))
	# active_config.computation_config = PlacefieldComputationParameters(speed_thresh=1, grid_bin=(10, 10), smooth=(0.5, 0.5)) ## Works well for 2D Placemaps

	# height: 20.0
	# width: 250.0
	active_config.computation_config = PlacefieldComputationParameters(speed_thresh=0, grid_bin=(2.0, 0.2), smooth=(0.5, 0.5)) ## Works well for 2D Placemaps


	## Compute Placefields if needed:
	try: active_epoch_placefields1D
	except NameError: active_epoch_placefields1D = None # Checks variable active_epoch_placefields's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
	try: active_epoch_placefields2D
	except NameError: active_epoch_placefields2D = None # Checks variable active_epoch_placefields's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
	# active_config.computation_config.smooth = (1.5, 0.5)
	active_epoch_placefields1D, active_epoch_placefields2D = compute_placefields_as_needed(active_epoch_session, active_config.computation_config, active_config, active_epoch_placefields1D, active_epoch_placefields2D, included_epochs=None, should_force_recompute_placefields=True, should_display_2D_plots=True)
	# Focus on the 2D placefields:
	active_epoch_placefields = active_epoch_placefields2D
	# Get the updated session using the units that have good placefields
	active_epoch_session, active_config, good_placefield_neuronIDs = process_by_good_placefields(active_epoch_session, active_config, active_epoch_placefields)
	# debug_print_spike_counts(active_epoch_session)
 
	try: active_lap_only_placefields1D
	except NameError: active_lap_only_placefields1D = None # Checks variable active_epoch_placefields's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
	try: active_lap_only_placefields2D
	except NameError: active_lap_only_placefields2D = None # Checks variable active_epoch_placefields's existance, and sets its value to None if it doesn't exist so it can be checked in the next step
	lap_specific_config = active_config
	lap_specific_included_epochs = Epoch(sess.laps.to_dataframe())
	# lap_specific_included_epochs = even_lap_specific_epochs # Kamran Right
	lap_specific_included_epochs = odd_lap_specific_epochs
	# active_lap_only_placefields1D, active_lap_only_placefields2D = compute_placefields_as_needed(active_lap_specific_epoch_session, lap_specific_config.computation_config, lap_specific_config, active_lap_only_placefields1D, active_lap_only_placefields2D, should_force_recompute_placefields=True, should_display_2D_plots=True)
	active_lap_only_placefields1D, active_lap_only_placefields2D = compute_placefields_as_needed(active_epoch_session, lap_specific_config.computation_config, lap_specific_config, active_lap_only_placefields1D, active_lap_only_placefields2D, included_epochs=lap_specific_included_epochs, should_force_recompute_placefields=True, should_display_2D_plots=True)

	# Focus on the 2D placefields:
	active_lap_only_placefields = active_lap_only_placefields2D
	# Get the updated session using the units that have good placefields
	lap_specific_session, lap_specific_config, lap_specific_good_placefield_neuronIDs = process_by_good_placefields(active_lap_specific_epoch_session, lap_specific_config, active_lap_only_placefields)