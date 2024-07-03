curr_active_pipeline.sess - printed by print_keys_if_possible on 2024-07-03
===================================================================================================


    curr_active_pipeline.sess: neuropy.core.session.dataSession.DataSession
	│   ├── config: neuropy.core.session.Formats.SessionSpecifications.SessionConfig
		│   ├── basepath: pathlib.WindowsPath
		│   ├── session_spec: neuropy.core.session.Formats.SessionSpecifications.SessionFolderSpec
			│   ├── required_files: list - (4,)
			│   ├── optional_files: list - (2,)
			│   ├── additional_validation_requirements: list - (0,)
		│   ├── session_name: str
		│   ├── session_context: neuropy.utils.result_context.IdentifyingContext
		│   ├── format_name: str
		│   ├── preprocessing_parameters: neuropy.core.session.Formats.SessionSpecifications.ParametersContainer
			│   ├── epoch_estimation_parameters: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── laps: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── N: int
					│   ├── should_backup_extant_laps_obj: bool
					│   ├── use_direction_dependent_laps: bool
				│   ├── PBEs: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── thresh: tuple - (2,)
					│   ├── min_dur: float
					│   ├── merge_dur: float
					│   ├── max_dur: float
				│   ├── replays: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── require_intersecting_epoch: neuropy.core.epoch.Epoch
						│   ├── _filename: NoneType
						│   ├── _metadata: NoneType
						│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (83, 4)
					│   ├── min_epoch_included_duration: float
					│   ├── max_epoch_included_duration: float
					│   ├── maximum_speed_thresh: NoneType
					│   ├── min_inclusion_fr_active_thresh: float
					│   ├── min_num_unique_aclu_inclusions: int
					│   ├── epochs_source: str
		│   ├── absolute_start_timestamp: float
		│   ├── position_sampling_rate_Hz: float
		│   ├── microseconds_to_seconds_conversion_factor: float
		│   ├── pix2cm: float
		│   ├── x_midpoint: float
		│   ├── loaded_track_limits: dict
			│   ├── long_xlim: numpy.ndarray - (2,)
			│   ├── short_xlim: numpy.ndarray - (2,)
			│   ├── long_ylim: numpy.ndarray - (2,)
			│   ├── short_ylim: numpy.ndarray - (2,)
		│   ├── is_resolved: bool
		│   ├── resolved_required_filespecs_dict: dict
			│   ├── W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.xml: neuropy.core.session.Formats.SessionSpecifications.SessionFileSpec
				│   ├── fileSpecString: str
				│   ├── suggestedBaseName: str
				│   ├── description: str
				│   ├── session_load_callback: method
			│   ├── W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.spikeII.mat: neuropy.core.session.Formats.SessionSpecifications.SessionFileSpec
				│   ├── fileSpecString: str
				│   ├── suggestedBaseName: str
				│   ├── description: str
				│   ├── session_load_callback: NoneType
			│   ├── W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.position_info.mat: neuropy.core.session.Formats.SessionSpecifications.SessionFileSpec
				│   ├── fileSpecString: str
				│   ├── suggestedBaseName: str
				│   ├── description: str
				│   ├── session_load_callback: NoneType
			│   ├── W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.epochs_info.mat: neuropy.core.session.Formats.SessionSpecifications.SessionFileSpec
				│   ├── fileSpecString: str
				│   ├── suggestedBaseName: str
				│   ├── description: str
				│   ├── session_load_callback: NoneType
		│   ├── resolved_optional_filespecs_dict: dict
			│   ├── W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.eeg: neuropy.core.session.Formats.SessionSpecifications.SessionFileSpec
				│   ├── fileSpecString: str
				│   ├── suggestedBaseName: str
				│   ├── description: str
				│   ├── session_load_callback: method
			│   ├── W:\Data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.dat: neuropy.core.session.Formats.SessionSpecifications.SessionFileSpec
				│   ├── fileSpecString: str
				│   ├── suggestedBaseName: str
				│   ├── description: str
				│   ├── session_load_callback: method
	│   ├── is_loaded: bool
	│   ├── filePrefix: pathlib.WindowsPath
	│   ├── recinfo: neuropy.io.neuroscopeio.NeuroscopeIO
		│   ├── source_file: pathlib.WindowsPath
		│   ├── eeg_filename: pathlib.WindowsPath
		│   ├── dat_filename: pathlib.WindowsPath
		│   ├── skipped_channels: numpy.ndarray - (0,)
		│   ├── channel_groups: numpy.ndarray - (9,)
		│   ├── discarded_channels: numpy.ndarray - (0,)
		│   ├── sig_dtype: int
		│   ├── dat_sampling_rate: int
		│   ├── eeg_sampling_rate: int
		│   ├── n_channels: int
		│   ├── good_channels: numpy.ndarray - (96,)
	│   ├── eegfile: neuropy.io.binarysignalio.BinarysignalIO
		│   ├── sampling_rate: int
		│   ├── n_channels: int
		│   ├── dtype: str
		│   ├── source_file: pathlib.WindowsPath
	│   ├── datfile: NoneType
	│   ├── neurons: neuropy.core.neurons.Neurons
		│   ├── _filename: NoneType
		│   ├── _metadata: NoneType
		│   ├── spiketrains: numpy.ndarray - (108,)
		│   ├── _neuron_ids: numpy.ndarray - (108,)
		│   ├── _reverse_cellID_index_map: dict (children omitted) - OMITTED TYPE WITH NO SHAPE
		│   ├── _extended_neuron_properties_df: NoneType
		│   ├── waveforms: NoneType
		│   ├── shank_ids: numpy.ndarray - (108,)
		│   ├── _neuron_type: numpy.ndarray - (108,)
		│   ├── peak_channels: NoneType
		│   ├── _sampling_rate: int
		│   ├── t_start: int
		│   ├── t_stop: numpy.float64
	│   ├── probegroup: NoneType
	│   ├── position: neuropy.core.position.Position
		│   ├── _filename: pathlib.WindowsPath
		│   ├── _metadata: NoneType
		│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (63192, 18)
	│   ├── paradigm: neuropy.core.epoch.Epoch
		│   ├── _filename: NoneType
		│   ├── _metadata: NoneType
		│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (2, 4)
	│   ├── ripple: neuropy.core.epoch.Epoch
		│   ├── _filename: pathlib.WindowsPath
		│   ├── _metadata: NoneType
		│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (6010, 6)
	│   ├── mua: neuropy.core.neurons.Mua
		│   ├── _filename: pathlib.WindowsPath
		│   ├── _metadata: NoneType
		│   ├── _spike_counts: numpy.ndarray - (2108138,)
		│   ├── t_start: int
		│   ├── _bin_size: float
	│   ├── laps: neuropy.core.laps.Laps
		│   ├── _filename: NoneType
		│   ├── _metadata: NoneType
		│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (82, 13)
	│   ├── flattened_spiketrains: neuropy.core.flattened_spiketrains.FlattenedSpiketrains
		│   ├── _filename: NoneType
		│   ├── _metadata: NoneType
		│   ├── _time_variable_name: str
		│   ├── _spikes_df: pandas.core.frame.DataFrame (children omitted) - (1014937, 25)
		│   ├── t_start: float
	│   ├── pbe: neuropy.core.epoch.Epoch
		│   ├── _filename: NoneType
		│   ├── _metadata: NoneType
		│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (634, 4)
	│   ├── replay: neuropy.core.epoch.Epoch
		│   ├── _filename: NoneType
		│   ├── _metadata: dict (children omitted)(all scalar values) - size: 1
		│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (766, 4)
	│   ├── laps_backup: neuropy.core.laps.Laps
		│   ├── _filename: NoneType
		│   ├── _metadata: NoneType
		│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (80, 17)
	│   ├── replay_backup: pandas.core.frame.DataFrame (children omitted) - (1426, 7)
