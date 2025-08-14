DirectionalMergedDecodersResult - printed by print_keys_if_possible on 2024-01-10
===================================================================================================


    DirectionalMergedDecodersResult: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions.DirectionalMergedDecodersResult
	│   ├── is_global: bool
	│   ├── all_directional_decoder_dict: dict
		│   ├── long_LR: neuropy.analyses.placefields.PfND
		│   ├── long_RL: neuropy.analyses.placefields.PfND
		│   ├── short_LR: neuropy.analyses.placefields.PfND
		│   ├── short_RL: neuropy.analyses.placefields.PfND
	│   ├── all_directional_pf1D_Decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
		│   ├── neuron_IDXs: numpy.ndarray - (68,)
		│   ├── neuron_IDs: numpy.ndarray - (68,)
		│   ├── F: numpy.ndarray - (232, 68)
		│   ├── P_x: numpy.ndarray - (232, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── long_directional_pf1D_Decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
		│   ├── neuron_IDXs: numpy.ndarray - (68,)
		│   ├── neuron_IDs: numpy.ndarray - (68,)
		│   ├── F: numpy.ndarray - (116, 68)
		│   ├── P_x: numpy.ndarray - (116, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── long_directional_decoder_dict: dict
		│   ├── long_LR: neuropy.analyses.placefields.PfND
		│   ├── long_RL: neuropy.analyses.placefields.PfND
	│   ├── short_directional_pf1D_Decoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder
		│   ├── pf: neuropy.analyses.placefields.PfND
		│   ├── neuron_IDXs: numpy.ndarray - (68,)
		│   ├── neuron_IDs: numpy.ndarray - (68,)
		│   ├── F: numpy.ndarray - (116, 68)
		│   ├── P_x: numpy.ndarray - (116, 1)
		│   ├── setup_on_init: bool
		│   ├── post_load_on_init: bool
		│   ├── debug_print: bool
	│   ├── short_directional_decoder_dict: dict
		│   ├── short_LR: neuropy.analyses.placefields.PfND
		│   ├── short_RL: neuropy.analyses.placefields.PfND
	│   ├── all_directional_laps_filter_epochs_decoder_result: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
		│   ├── decoding_time_bin_size: float
		│   ├── filter_epochs: neuropy.core.epoch.Epoch
		│   ├── num_filter_epochs: int
		│   ├── most_likely_positions_list: list - (88,)
		│   ├── p_x_given_n_list: list - 88
		│   ├── marginal_x_list: list - (88, 2)
		│   ├── marginal_y_list: list - (88, 2)
		│   ├── most_likely_position_indicies_list: list - 88
		│   ├── spkcount: list - 88
		│   ├── nbins: numpy.ndarray - (88,)
		│   ├── time_bin_containers: list - (88,)
		│   ├── time_bin_edges: list - (88,)
		│   ├── epoch_description_list: list - (0,)
	│   ├── all_directional_ripple_filter_epochs_decoder_result: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.DecodedFilterEpochsResult
		│   ├── decoding_time_bin_size: float
		│   ├── filter_epochs: pandas.core.frame.DataFrame (children omitted) - (707, 4)
		│   ├── num_filter_epochs: int
		│   ├── most_likely_positions_list: list - (707,)
		│   ├── p_x_given_n_list: list - 707
		│   ├── marginal_x_list: list - (707, 2)
		│   ├── marginal_y_list: list - (707, 2)
		│   ├── most_likely_position_indicies_list: list - 707
		│   ├── spkcount: list - 707
		│   ├── nbins: numpy.ndarray - (707,)
		│   ├── time_bin_containers: list - (707,)
		│   ├── time_bin_edges: list - (707,)
		│   ├── epoch_description_list: list - (0,)
	│   ├── laps_directional_marginals_tuple: tuple - 4
	│   ├── laps_track_identity_marginals_tuple: tuple - 4
	│   ├── ripple_directional_marginals_tuple: tuple - 4
	│   ├── ripple_track_identity_marginals_tuple: tuple - 4
