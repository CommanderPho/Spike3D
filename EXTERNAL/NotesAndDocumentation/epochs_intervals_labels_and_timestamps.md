


epochs_df: L epochs

dataseries_timestamps_array




compute_PBEs_spikes_df(spk_df, pbe_epoch_df):
        """ Adds a 'PBE_id' column to the spikes_df:

# Situation 1: 
Assign the appropriate epoch_IDX to each spike in spk_df
	- If the epochs are non-overlapping, each spike will have either:
		!- No Epoch (epoch_IDX .= -1)
		!- Exactly 1 Epoch (epoch_IDX .= epoch_IDX)

Performed using:
	 _compute_spike_arbitrary_provided_epoch_ids(spk_df, provided_epochs_df, epoch_label_column_name=None, no_interval_fill_value=np.nan, override_time_variable_name=None, debug_print=False)
	

### Examples:
	add_PBE_identity(spk_df, pbe_epoch_df, no_interval_fill_value=np.nan)
		- Adds the PBE identity to the spikes_df
	add_epochs_id_identity(spk_df, epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name='label', override_time_variable_name=None, no_interval_fill_value=np.nan)
		- Adds the epoch IDs to each spike in spikes_df as a column named epoch_id_key_name






def _searchsorted_find_event_interval_indicies(times_arr, start_stop_times_arr): # Function is compiled by numba and runs in machine code
    """Converts the L x 2 array of start and stop times (start_stop_times_arr) representing intervals in time to an array of indicies into times_arr of the same size

    Args:
        times_arr (np.ndarray): An array of times of shape (N, ) in the same units as the start_stop_times_arr
        start_stop_times_arr (np.ndarray): An array of start and stop intervals of shape (L, 2), with start_stop_times_arr[:, 0] representing the start times and start_stop_times_arr[:, 1] representing the stop times.

    Returns:
        np.ndarray: An array of start and stop indicies into times_arr of shape (L, 2)





# Core helpers:
determine_event_interval_identity(...)

https://github.com/CommanderPho/NeuroPy/blob/main/neuropy/utils/efficient_interval_search.py#L182

https://github.com/CommanderPho/NeuroPy/blob/main/neuropy/utils/efficient_interval_search.py#L190

