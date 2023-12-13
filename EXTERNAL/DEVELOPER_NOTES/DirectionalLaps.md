DirectionalLaps - printed by print_keys_if_possible on 2023-12-13
===================================================================================================


    DirectionalLaps: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalLapsResult
	│   ├── is_global: bool
	│   ├── directional_lap_specific_configs: dict
		│   ├── maze1_odd: neuropy.utils.dynamic_container.DynamicContainer
			│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
				│   ├── time_bin_size: float
				│   ├── computation_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
			│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── max_num_spikes_per_neuron: int
				│   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── s: int
					│   ├── gamma: float
				│   ├── use_progress_bar: bool
				│   ├── debug_print: bool
		│   ├── maze1_even: neuropy.utils.dynamic_container.DynamicContainer
			│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
				│   ├── time_bin_size: float
				│   ├── computation_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
			│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── max_num_spikes_per_neuron: int
				│   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── s: int
					│   ├── gamma: float
				│   ├── use_progress_bar: bool
				│   ├── debug_print: bool
		│   ├── maze2_odd: neuropy.utils.dynamic_container.DynamicContainer
			│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
				│   ├── time_bin_size: float
				│   ├── computation_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
			│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── max_num_spikes_per_neuron: int
				│   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── s: int
					│   ├── gamma: float
				│   ├── use_progress_bar: bool
				│   ├── debug_print: bool
		│   ├── maze2_even: neuropy.utils.dynamic_container.DynamicContainer
			│   ├── pf_params: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
				│   ├── time_bin_size: float
				│   ├── computation_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
			│   ├── spike_analysis: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── max_num_spikes_per_neuron: int
				│   ├── kleinberg_parameters: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── s: int
					│   ├── gamma: float
				│   ├── use_progress_bar: bool
				│   ├── debug_print: bool
	│   ├── split_directional_laps_dict: dict
		│   ├── maze1_odd: neuropy.core.epoch.Epoch
			│   ├── _filename: NoneType
			│   ├── _metadata: NoneType
			│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
		│   ├── maze1_even: neuropy.core.epoch.Epoch
			│   ├── _filename: NoneType
			│   ├── _metadata: NoneType
			│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
		│   ├── maze2_odd: neuropy.core.epoch.Epoch
			│   ├── _filename: NoneType
			│   ├── _metadata: NoneType
			│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
		│   ├── maze2_even: neuropy.core.epoch.Epoch
			│   ├── _filename: NoneType
			│   ├── _metadata: NoneType
			│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
	│   ├── split_directional_laps_contexts_dict: dict
		│   ├── maze1_odd: neuropy.utils.result_context.IdentifyingContext
		│   ├── maze1_even: neuropy.utils.result_context.IdentifyingContext
		│   ├── maze2_odd: neuropy.utils.result_context.IdentifyingContext
		│   ├── maze2_even: neuropy.utils.result_context.IdentifyingContext
	│   ├── split_directional_laps_config_names: list - (4,)
	│   ├── computed_base_epoch_names: list - (0,)
	│   ├── long_LR_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (87501, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (30844, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (55, 56)
				│   ├── tuning_curves: numpy.ndarray - (55, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (55, 56)
				│   ├── _neuron_ids: list - (55,)
				│   ├── _neuron_extended_ids: list - (55, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (55,)
			│   ├── _ratemap_spiketrains_pos: list - (55, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3947, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (17326, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (55,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (55,)
		│   ├── neuron_IDs: numpy.ndarray - (55,)
		│   ├── F: numpy.ndarray - (56, 55)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── long_RL_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (87501, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (30844, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (59, 56)
				│   ├── tuning_curves: numpy.ndarray - (59, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (59, 56)
				│   ├── _neuron_ids: list - (59,)
				│   ├── _neuron_extended_ids: list - (59, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (59,)
			│   ├── _ratemap_spiketrains_pos: list - (59, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (4171, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (14855, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (59,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (59,)
		│   ├── neuron_IDs: numpy.ndarray - (59,)
		│   ├── F: numpy.ndarray - (56, 59)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── short_LR_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (62184, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (21212, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (55, 56)
				│   ├── tuning_curves: numpy.ndarray - (55, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (55, 56)
				│   ├── _neuron_ids: list - (55,)
				│   ├── _neuron_extended_ids: list - (55, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (55,)
			│   ├── _ratemap_spiketrains_pos: list - (55, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (2649, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (11653, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (55,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (55,)
		│   ├── neuron_IDs: numpy.ndarray - (55,)
		│   ├── F: numpy.ndarray - (56, 55)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── short_RL_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (62184, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (21212, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (59, 56)
				│   ├── tuning_curves: numpy.ndarray - (59, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (59, 56)
				│   ├── _neuron_ids: list - (59,)
				│   ├── _neuron_extended_ids: list - (59, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (59,)
			│   ├── _ratemap_spiketrains_pos: list - (59, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3092, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (11549, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (59,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (59,)
		│   ├── neuron_IDs: numpy.ndarray - (59,)
		│   ├── F: numpy.ndarray - (56, 59)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── long_LR_shared_aclus_only_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (87501, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (30844, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (55, 56)
				│   ├── tuning_curves: numpy.ndarray - (55, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (55, 56)
				│   ├── _neuron_ids: list - (55,)
				│   ├── _neuron_extended_ids: list - (55, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (55,)
			│   ├── _ratemap_spiketrains_pos: list - (55, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3947, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (17326, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (55,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (55,)
		│   ├── neuron_IDs: numpy.ndarray - (55,)
		│   ├── F: numpy.ndarray - (56, 55)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── long_RL_shared_aclus_only_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (87501, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (30844, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (59, 56)
				│   ├── tuning_curves: numpy.ndarray - (59, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (59, 56)
				│   ├── _neuron_ids: list - (59,)
				│   ├── _neuron_extended_ids: list - (59, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (59,)
			│   ├── _ratemap_spiketrains_pos: list - (59, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (4171, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (14855, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (59,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (59,)
		│   ├── neuron_IDs: numpy.ndarray - (59,)
		│   ├── F: numpy.ndarray - (56, 59)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── short_LR_shared_aclus_only_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (62184, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (21212, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (55, 56)
				│   ├── tuning_curves: numpy.ndarray - (55, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (55, 56)
				│   ├── _neuron_ids: list - (55,)
				│   ├── _neuron_extended_ids: list - (55, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (55,)
			│   ├── _ratemap_spiketrains_pos: list - (55, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (2649, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (11653, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (55,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (55,)
		│   ├── neuron_IDs: numpy.ndarray - (55,)
		│   ├── F: numpy.ndarray - (56, 55)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── short_RL_shared_aclus_only_one_step_decoder_1D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (62184, 25)
			│   ├── position: neuropy.core.position.Position
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (21212, 10)
			│   ├── epochs: neuropy.core.epoch.Epoch
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (20, 6)
			│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters
				│   ├── speed_thresh: float
				│   ├── grid_bin: tuple - (2,)
				│   ├── grid_bin_bounds: tuple - (2, 2)
				│   ├── smooth: tuple - (2,)
				│   ├── frate_thresh: float
				│   ├── is_directional: bool
			│   ├── position_srate: numpy.float64
			│   ├── setup_on_init: bool
			│   ├── compute_on_init: bool
			│   ├── _save_intermediate_spikes_maps: bool
			│   ├── _ratemap: neuropy.core.ratemap.Ratemap
				│   ├── _filename: NoneType
				│   ├── _metadata: NoneType
				│   ├── spikes_maps: numpy.ndarray - (59, 56)
				│   ├── tuning_curves: numpy.ndarray - (59, 56)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (59, 56)
				│   ├── _neuron_ids: list - (59,)
				│   ├── _neuron_extended_ids: list - (59, 3)
				│   ├── xbin: numpy.ndarray - (57,)
				│   ├── ybin: NoneType
				│   ├── occupancy: numpy.ndarray - (56,)
			│   ├── _ratemap_spiketrains: list - (59,)
			│   ├── _ratemap_spiketrains_pos: list - (59, 1)
			│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3092, 11)
			│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (11549, 25)
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (57,)
			│   ├── ybin: NoneType
			│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
			│   ├── _included_thresh_neurons_indx: list - (59,)
			│   ├── _peak_frate_filter_function: function
		│   ├── neuron_IDXs: numpy.ndarray - (59,)
		│   ├── neuron_IDs: numpy.ndarray - (59,)
		│   ├── F: numpy.ndarray - (56, 59)
		│   ├── P_x: numpy.ndarray - (56, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
