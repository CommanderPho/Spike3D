InteractivePlaceCellConfig - printed by print_keys_if_possible on 2023-10-23
===================================================================================================


    InteractivePlaceCellConfig: pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.InteractivePlaceCellConfig
	│   ├── active_session_config: neuropy.core.session.Formats.SessionSpecifications.SessionConfig
		│   ├── format_name: str
		│   ├── basepath: pathlib.WindowsPath
		│   ├── session_name: str
		│   ├── session_spec: neuropy.core.session.Formats.SessionSpecifications.SessionFolderSpec
			│   ├── required_files: list - (4,)
			│   ├── optional_files: list - (2,)
			│   ├── additional_validation_requirements: list - (0,)
		│   ├── session_context: neuropy.utils.result_context.IdentifyingContext
		│   ├── preprocessing_parameters: neuropy.core.session.Formats.SessionSpecifications.ParametersContainer
			│   ├── epoch_estimation_parameters: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── laps: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── N: int
					│   ├── should_backup_extant_laps_obj: bool
				│   ├── PBEs: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── thresh: tuple - (2,)
					│   ├── min_dur: float
					│   ├── merge_dur: float
					│   ├── max_dur: float
				│   ├── replays: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── require_intersecting_epoch: neuropy.core.epoch.Epoch
						│   ├── _filename: NoneType
						│   ├── _metadata: NoneType
						│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (81, 4)
					│   ├── min_epoch_included_duration: float
					│   ├── max_epoch_included_duration: float
					│   ├── maximum_speed_thresh: NoneType
					│   ├── min_inclusion_fr_active_thresh: float
					│   ├── min_num_unique_aclu_inclusions: int
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
		│   ├── absolute_start_timestamp: float
		│   ├── position_sampling_rate_Hz: float
	│   ├── active_epochs: neuropy.core.epoch.NamedTimerange
		│   ├── name: str
		│   ├── start_end_times: list - (2,)
	│   ├── video_output_config: pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.VideoOutputModeConfig
		│   ├── active_frame_range: numpy.ndarray - (20,)
		│   ├── video_output_parent_dir: pathlib.WindowsPath
		│   ├── active_is_video_output_mode: bool
		│   ├── active_video_output_parent_dir: pathlib.WindowsPath
		│   ├── active_video_output_filename: str
		│   ├── active_video_output_fullpath: pathlib.WindowsPath
	│   ├── plotting_config: pyphoplacecellanalysis.General.Model.Configs.DynamicConfigs.PlottingConfig
		│   ├── output_subplots_shape: tuple - (2,)
		│   ├── output_parent_dir: pathlib.WindowsPath
		│   ├── use_age_proportional_spike_scale: bool
		│   ├── plotter_type: str
		│   ├── subplots_shape: tuple - (2,)
		│   ├── active_output_parent_dir: pathlib.WindowsPath
		│   ├── pf_neuron_identities: list - (81,)
		│   ├── pf_sort_ind: numpy.ndarray - (81,)
		│   ├── pf_colors: numpy.ndarray - (4, 81)
		│   ├── pf_colormap: numpy.ndarray - (81, 4)
		│   ├── pf_listed_colormap: matplotlib.colors.ListedColormap
			│   ├── monochrome: bool
			│   ├── colors: numpy.ndarray - (81, 4)
			│   ├── name: str
			│   ├── N: int
			│   ├── _rgba_bad: tuple - (4,)
			│   ├── _rgba_under: NoneType
			│   ├── _rgba_over: NoneType
			│   ├── _i_under: int
			│   ├── _i_over: int
			│   ├── _i_bad: int
			│   ├── _isinit: bool
			│   ├── colorbar_extend: bool
		│   ├── use_smoothed_maze_rendering: bool
	│   ├── computation_config: neuropy.utils.dynamic_container.DynamicContainer
		│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
			│   ├── speed_thresh: float
			│   ├── grid_bin: tuple - (2,)
			│   ├── grid_bin_bounds: tuple - (2, 2)
			│   ├── smooth: tuple - (2,)
			│   ├── frate_thresh: float
			│   ├── time_bin_size: float
			│   ├── computation_epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (40, 6)
		│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
			│   ├── max_num_spikes_per_neuron: int
			│   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── s: int
				│   ├── gamma: float
			│   ├── use_progress_bar: bool
			│   ├── debug_print: bool
	│   ├── filter_config: dict
		│   ├── filter_function: function
