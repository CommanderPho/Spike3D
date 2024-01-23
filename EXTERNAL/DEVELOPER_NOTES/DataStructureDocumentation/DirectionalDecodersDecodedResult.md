DirectionalDecodersDecodedResult - printed by print_keys_if_possible on 2024-01-22
===================================================================================================


    DirectionalDecodersDecodedResult: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalDecodersDecodedResult
	│   ├── is_global: bool
	│   ├── result_version: str
	│   ├── _VersionedResultMixin_version: str
	│   ├── pf1D_Decoder_dict: dict
		│   ├── long_LR: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder
			│   ├── pf: neuropy.analyses.placefields.PfND
				│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (85775, 25)
				│   ├── position: neuropy.core.position.Position
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (36310, 10)
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
					│   ├── spikes_maps: numpy.ndarray - (62, 62)
					│   ├── tuning_curves: numpy.ndarray - (62, 62)
					│   ├── unsmoothed_tuning_maps: numpy.ndarray - (62, 62)
					│   ├── _neuron_ids: list - (62,)
					│   ├── _neuron_extended_ids: list - (62, 3)
					│   ├── xbin: numpy.ndarray - (63,)
					│   ├── ybin: NoneType
					│   ├── occupancy: numpy.ndarray - (62,)
				│   ├── _ratemap_spiketrains: list - (62,)
				│   ├── _ratemap_spiketrains_pos: list - (62, 1)
				│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3866, 11)
				│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (15853, 25)
				│   ├── ndim: numpy.int32
				│   ├── xbin: numpy.ndarray - (63,)
				│   ├── ybin: NoneType
				│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
				│   ├── _included_thresh_neurons_indx: NoneType
				│   ├── _peak_frate_filter_function: NoneType
			│   ├── neuron_IDXs: numpy.ndarray - (62,)
			│   ├── neuron_IDs: numpy.ndarray - (62,)
			│   ├── F: numpy.ndarray - (62, 62)
			│   ├── P_x: numpy.ndarray - (62, 1)
			│   ├── setup_on_init: bool
			│   ├── post_load_on_init: bool
			│   ├── debug_print: bool
			│   ├── time_bin_size: float
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (15853, 26)
			│   ├── time_binning_container: neuropy.utils.mixins.binning_helpers.BinningContainer
				│   ├── edges: numpy.ndarray - (35509,)
				│   ├── centers: numpy.ndarray - (35508,)
				│   ├── edge_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (35509,)
				│   ├── center_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (35508,)
			│   ├── unit_specific_time_binned_spike_counts: numpy.ndarray - (62, 35157)
			│   ├── total_spike_counts_per_window: numpy.ndarray - (35157,)
			│   ├── flat_p_x_given_n: numpy.ndarray - (62, 35508)
			│   ├── p_x_given_n: numpy.ndarray - (62, 35508)
			│   ├── most_likely_position_flat_indicies: numpy.ndarray - (35508,)
			│   ├── most_likely_position_indicies: NoneType
			│   ├── marginal: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── x: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── p_x_given_n: numpy.ndarray - (62, 35508)
					│   ├── most_likely_positions_1D: numpy.ndarray - (35508,)
					│   ├── revised_most_likely_positions_1D: numpy.ndarray - (35508,)
				│   ├── y: NoneType
			│   ├── most_likely_positions: numpy.ndarray - (35508,)
			│   ├── revised_most_likely_positions: numpy.ndarray - (35508,)
		│   ├── long_RL: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder
			│   ├── pf: neuropy.analyses.placefields.PfND
				│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (85775, 25)
				│   ├── position: neuropy.core.position.Position
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (36310, 10)
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
					│   ├── spikes_maps: numpy.ndarray - (64, 62)
					│   ├── tuning_curves: numpy.ndarray - (64, 62)
					│   ├── unsmoothed_tuning_maps: numpy.ndarray - (64, 62)
					│   ├── _neuron_ids: list - (64,)
					│   ├── _neuron_extended_ids: list - (64, 3)
					│   ├── xbin: numpy.ndarray - (63,)
					│   ├── ybin: NoneType
					│   ├── occupancy: numpy.ndarray - (62,)
				│   ├── _ratemap_spiketrains: list - (64,)
				│   ├── _ratemap_spiketrains_pos: list - (64, 1)
				│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (4806, 11)
				│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (13683, 25)
				│   ├── ndim: numpy.int32
				│   ├── xbin: numpy.ndarray - (63,)
				│   ├── ybin: NoneType
				│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
				│   ├── _included_thresh_neurons_indx: NoneType
				│   ├── _peak_frate_filter_function: NoneType
			│   ├── neuron_IDXs: numpy.ndarray - (64,)
			│   ├── neuron_IDs: numpy.ndarray - (64,)
			│   ├── F: numpy.ndarray - (62, 64)
			│   ├── P_x: numpy.ndarray - (62, 1)
			│   ├── setup_on_init: bool
			│   ├── post_load_on_init: bool
			│   ├── debug_print: bool
			│   ├── time_bin_size: float
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (13683, 26)
			│   ├── time_binning_container: neuropy.utils.mixins.binning_helpers.BinningContainer
				│   ├── edges: numpy.ndarray - (35805,)
				│   ├── centers: numpy.ndarray - (35804,)
				│   ├── edge_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (35805,)
				│   ├── center_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (35804,)
			│   ├── unit_specific_time_binned_spike_counts: numpy.ndarray - (64, 35451)
			│   ├── total_spike_counts_per_window: numpy.ndarray - (35451,)
			│   ├── flat_p_x_given_n: numpy.ndarray - (62, 35804)
			│   ├── p_x_given_n: numpy.ndarray - (62, 35804)
			│   ├── most_likely_position_flat_indicies: numpy.ndarray - (35804,)
			│   ├── most_likely_position_indicies: NoneType
			│   ├── marginal: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── x: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── p_x_given_n: numpy.ndarray - (62, 35804)
					│   ├── most_likely_positions_1D: numpy.ndarray - (35804,)
					│   ├── revised_most_likely_positions_1D: numpy.ndarray - (35804,)
				│   ├── y: NoneType
			│   ├── most_likely_positions: numpy.ndarray - (35804,)
			│   ├── revised_most_likely_positions: numpy.ndarray - (35804,)
		│   ├── short_LR: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder
			│   ├── pf: neuropy.analyses.placefields.PfND
				│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (65180, 25)
				│   ├── position: neuropy.core.position.Position
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (26443, 10)
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
					│   ├── spikes_maps: numpy.ndarray - (62, 62)
					│   ├── tuning_curves: numpy.ndarray - (62, 62)
					│   ├── unsmoothed_tuning_maps: numpy.ndarray - (62, 62)
					│   ├── _neuron_ids: list - (62,)
					│   ├── _neuron_extended_ids: list - (62, 3)
					│   ├── xbin: numpy.ndarray - (63,)
					│   ├── ybin: NoneType
					│   ├── occupancy: numpy.ndarray - (62,)
				│   ├── _ratemap_spiketrains: list - (62,)
				│   ├── _ratemap_spiketrains_pos: list - (62, 1)
				│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3530, 11)
				│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (12807, 25)
				│   ├── ndim: numpy.int32
				│   ├── xbin: numpy.ndarray - (63,)
				│   ├── ybin: NoneType
				│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
				│   ├── _included_thresh_neurons_indx: NoneType
				│   ├── _peak_frate_filter_function: NoneType
			│   ├── neuron_IDXs: numpy.ndarray - (62,)
			│   ├── neuron_IDs: numpy.ndarray - (62,)
			│   ├── F: numpy.ndarray - (62, 62)
			│   ├── P_x: numpy.ndarray - (62, 1)
			│   ├── setup_on_init: bool
			│   ├── post_load_on_init: bool
			│   ├── debug_print: bool
			│   ├── time_bin_size: float
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (12807, 26)
			│   ├── time_binning_container: neuropy.utils.mixins.binning_helpers.BinningContainer
				│   ├── edges: numpy.ndarray - (24560,)
				│   ├── centers: numpy.ndarray - (24559,)
				│   ├── edge_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (24560,)
				│   ├── center_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (24559,)
			│   ├── unit_specific_time_binned_spike_counts: numpy.ndarray - (62, 24317)
			│   ├── total_spike_counts_per_window: numpy.ndarray - (24317,)
			│   ├── flat_p_x_given_n: numpy.ndarray - (62, 24559)
			│   ├── p_x_given_n: numpy.ndarray - (62, 24559)
			│   ├── most_likely_position_flat_indicies: numpy.ndarray - (24559,)
			│   ├── most_likely_position_indicies: NoneType
			│   ├── marginal: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── x: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── p_x_given_n: numpy.ndarray - (62, 24559)
					│   ├── most_likely_positions_1D: numpy.ndarray - (24559,)
					│   ├── revised_most_likely_positions_1D: numpy.ndarray - (24559,)
				│   ├── y: NoneType
			│   ├── most_likely_positions: numpy.ndarray - (24559,)
			│   ├── revised_most_likely_positions: numpy.ndarray - (24559,)
		│   ├── short_RL: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder
			│   ├── pf: neuropy.analyses.placefields.PfND
				│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (65180, 25)
				│   ├── position: neuropy.core.position.Position
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (26443, 10)
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
					│   ├── spikes_maps: numpy.ndarray - (63, 62)
					│   ├── tuning_curves: numpy.ndarray - (63, 62)
					│   ├── unsmoothed_tuning_maps: numpy.ndarray - (63, 62)
					│   ├── _neuron_ids: list - (63,)
					│   ├── _neuron_extended_ids: list - (63, 3)
					│   ├── xbin: numpy.ndarray - (63,)
					│   ├── ybin: NoneType
					│   ├── occupancy: numpy.ndarray - (62,)
				│   ├── _ratemap_spiketrains: list - (63,)
				│   ├── _ratemap_spiketrains_pos: list - (63, 1)
				│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (3579, 11)
				│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (11557, 25)
				│   ├── ndim: numpy.int32
				│   ├── xbin: numpy.ndarray - (63,)
				│   ├── ybin: NoneType
				│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
				│   ├── _included_thresh_neurons_indx: NoneType
				│   ├── _peak_frate_filter_function: NoneType
			│   ├── neuron_IDXs: numpy.ndarray - (63,)
			│   ├── neuron_IDs: numpy.ndarray - (63,)
			│   ├── F: numpy.ndarray - (62, 63)
			│   ├── P_x: numpy.ndarray - (62, 1)
			│   ├── setup_on_init: bool
			│   ├── post_load_on_init: bool
			│   ├── debug_print: bool
			│   ├── time_bin_size: float
			│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (11557, 26)
			│   ├── time_binning_container: neuropy.utils.mixins.binning_helpers.BinningContainer
				│   ├── edges: numpy.ndarray - (24377,)
				│   ├── centers: numpy.ndarray - (24376,)
				│   ├── edge_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (24377,)
				│   ├── center_info: neuropy.utils.mixins.binning_helpers.BinningInfo
					│   ├── variable_extents: list - (2,)
					│   ├── step: numpy.float64
					│   ├── num_bins: int
					│   ├── bin_indicies: numpy.ndarray - (24376,)
			│   ├── unit_specific_time_binned_spike_counts: numpy.ndarray - (63, 24136)
			│   ├── total_spike_counts_per_window: numpy.ndarray - (24136,)
			│   ├── flat_p_x_given_n: numpy.ndarray - (62, 24376)
			│   ├── p_x_given_n: numpy.ndarray - (62, 24376)
			│   ├── most_likely_position_flat_indicies: numpy.ndarray - (24376,)
			│   ├── most_likely_position_indicies: NoneType
			│   ├── marginal: neuropy.utils.dynamic_container.DynamicContainer
				│   ├── x: neuropy.utils.dynamic_container.DynamicContainer
					│   ├── p_x_given_n: numpy.ndarray - (62, 24376)
					│   ├── most_likely_positions_1D: numpy.ndarray - (24376,)
					│   ├── revised_most_likely_positions_1D: numpy.ndarray - (24376,)
				│   ├── y: NoneType
			│   ├── most_likely_positions: numpy.ndarray - (24376,)
			│   ├── revised_most_likely_positions: numpy.ndarray - (24376,)
	│   ├── pseudo2D_decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
			│   ├── spikes_df: NoneType
			│   ├── position: NoneType
			│   ├── epochs: NoneType
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
				│   ├── spikes_maps: numpy.ndarray - (80, 62, 4)
				│   ├── tuning_curves: numpy.ndarray - (80, 62, 4)
				│   ├── unsmoothed_tuning_maps: numpy.ndarray - (80, 62, 4)
				│   ├── _neuron_ids: numpy.ndarray - (80,)
				│   ├── _neuron_extended_ids: list - (80, 3)
				│   ├── xbin: numpy.ndarray - (63,)
				│   ├── ybin: numpy.ndarray - (5,)
				│   ├── occupancy: numpy.ndarray - (62, 4)
			│   ├── _ratemap_spiketrains: NoneType
			│   ├── _ratemap_spiketrains_pos: NoneType
			│   ├── _filtered_pos_df: NoneType
			│   ├── _filtered_spikes_df: NoneType
			│   ├── ndim: numpy.int32
			│   ├── xbin: numpy.ndarray - (63,)
			│   ├── ybin: numpy.ndarray - (5,)
			│   ├── bin_info: NoneType
			│   ├── _included_thresh_neurons_indx: NoneType
			│   ├── _peak_frate_filter_function: NoneType
		│   ├── neuron_IDXs: numpy.ndarray - (80,)
		│   ├── neuron_IDs: numpy.ndarray - (80,)
		│   ├── F: numpy.ndarray - (248, 80)
		│   ├── P_x: numpy.ndarray - (248, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (142306, 25)
	│   ├── continuously_decoded_result_cache_dict: dict
		│   ├── 0.01: dict
			│   ├── long_LR: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
				│   ├── decoding_time_bin_size: float
				│   ├── filter_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (1, 4)
				│   ├── num_filter_epochs: int
				│   ├── most_likely_positions_list: list - (1, 209389)
				│   ├── p_x_given_n_list: list - (1, 62, 209389)
				│   ├── marginal_x_list: list - (1, 2)
				│   ├── marginal_y_list: list - (1,)
				│   ├── most_likely_position_indicies_list: list - (1, 1, 209389)
				│   ├── spkcount: list - (1, 62, 209389)
				│   ├── nbins: numpy.ndarray - (1,)
				│   ├── time_bin_containers: list - (1,)
				│   ├── time_bin_edges: list - (1, 209390)
				│   ├── epoch_description_list: list - (0,)
			│   ├── long_RL: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
				│   ├── decoding_time_bin_size: float
				│   ├── filter_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (1, 4)
				│   ├── num_filter_epochs: int
				│   ├── most_likely_positions_list: list - (1, 209389)
				│   ├── p_x_given_n_list: list - (1, 62, 209389)
				│   ├── marginal_x_list: list - (1, 2)
				│   ├── marginal_y_list: list - (1,)
				│   ├── most_likely_position_indicies_list: list - (1, 1, 209389)
				│   ├── spkcount: list - (1, 64, 209389)
				│   ├── nbins: numpy.ndarray - (1,)
				│   ├── time_bin_containers: list - (1,)
				│   ├── time_bin_edges: list - (1, 209390)
				│   ├── epoch_description_list: list - (0,)
			│   ├── short_LR: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
				│   ├── decoding_time_bin_size: float
				│   ├── filter_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (1, 4)
				│   ├── num_filter_epochs: int
				│   ├── most_likely_positions_list: list - (1, 209389)
				│   ├── p_x_given_n_list: list - (1, 62, 209389)
				│   ├── marginal_x_list: list - (1, 2)
				│   ├── marginal_y_list: list - (1,)
				│   ├── most_likely_position_indicies_list: list - (1, 1, 209389)
				│   ├── spkcount: list - (1, 62, 209389)
				│   ├── nbins: numpy.ndarray - (1,)
				│   ├── time_bin_containers: list - (1,)
				│   ├── time_bin_edges: list - (1, 209390)
				│   ├── epoch_description_list: list - (0,)
			│   ├── short_RL: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
				│   ├── decoding_time_bin_size: float
				│   ├── filter_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (1, 4)
				│   ├── num_filter_epochs: int
				│   ├── most_likely_positions_list: list - (1, 209389)
				│   ├── p_x_given_n_list: list - (1, 62, 209389)
				│   ├── marginal_x_list: list - (1, 2)
				│   ├── marginal_y_list: list - (1,)
				│   ├── most_likely_position_indicies_list: list - (1, 1, 209389)
				│   ├── spkcount: list - (1, 63, 209389)
				│   ├── nbins: numpy.ndarray - (1,)
				│   ├── time_bin_containers: list - (1,)
				│   ├── time_bin_edges: list - (1, 209390)
				│   ├── epoch_description_list: list - (0,)
			│   ├── pseudo2D: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
				│   ├── decoding_time_bin_size: float
				│   ├── filter_epochs: neuropy.core.epoch.Epoch
					│   ├── _filename: NoneType
					│   ├── _metadata: NoneType
					│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (1, 4)
				│   ├── num_filter_epochs: int
				│   ├── most_likely_positions_list: list - (1, 209389, 2)
				│   ├── p_x_given_n_list: list - (1, 62, 4, 209389)
				│   ├── marginal_x_list: list - (1, 2)
				│   ├── marginal_y_list: list - (1, 2)
				│   ├── most_likely_position_indicies_list: list - (1, 2, 209389)
				│   ├── spkcount: list - (1, 80, 209389)
				│   ├── nbins: numpy.ndarray - (1,)
				│   ├── time_bin_containers: list - (1,)
				│   ├── time_bin_edges: list - (1, 209390)
				│   ├── epoch_description_list: list - (0,)
