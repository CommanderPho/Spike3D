SessionConfig - printed by print_keys_if_possible on 2024-06-26
===================================================================================================


    SessionConfig: neuropy.core.session.Formats.SessionSpecifications.SessionConfig
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
