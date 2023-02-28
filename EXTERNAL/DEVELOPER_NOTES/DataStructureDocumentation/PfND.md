PfND - printed by print_keys_if_possible on 2023-02-27
===================================================================================================


- PfND: neuropy.analyses.placefields.PfND
	- config: neuropy.analyses.placefields.PlacefieldComputationParameters
		- speed_thresh: float
		- grid_bin: tuple - (2,)
		- grid_bin_bounds: tuple - (2,)
		- smooth: tuple - (2,)
		- frate_thresh: float
	- position_srate: numpy.float64
	- ndim: numpy.int32
	- xbin: numpy.ndarray - (64,)
	- ybin: NoneType
	- bin_info: dict
		- mode: str
		- xstep: numpy.float64
		- xnum_bins: int
	- _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (182109, 22)
	- _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (36310, 5)
	- ratemap: neuropy.core.ratemap.Ratemap
		- _filename: NoneType
		- _metadata: NoneType
		- spikes_maps: numpy.ndarray - (104, 63)
		- tuning_curves: numpy.ndarray - (104, 63)
		- unsmoothed_tuning_maps: numpy.ndarray - (104, 63)
		- _neuron_ids: list - (104,)
		- _neuron_extended_ids: list - (104, 3)
		- xbin: numpy.ndarray - (64,)
		- ybin: NoneType
		- occupancy: numpy.ndarray - (63,)
	- ratemap_spiketrains: list - (104,)
	- ratemap_spiketrains_pos: list - (104, 1)
	- _save_intermediate_spikes_maps: bool
	- _included_thresh_neurons_indx: NoneType
	- _peak_frate_filter_function: NoneType
