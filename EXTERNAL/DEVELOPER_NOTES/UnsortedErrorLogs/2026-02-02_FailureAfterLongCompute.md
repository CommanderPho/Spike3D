# Cell executed:
```python
# active_epoch_names = epoch_names
active_epoch_names = ['roam', 'sprinkle' ]
fine_decoding_time_bin_size: float = 0.025
for an_epoch_name in active_epoch_names:    
    try:
        print(f'\ttrying `.masked_container.final_refine_single_epoch_result_masks(...)` for an_epoch_name: "{an_epoch_name}"...')
        if an_epoch_name not in masked_container.debug_computed_dict:
            masked_container.debug_computed_dict[an_epoch_name] = {}
        
        # active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = a_masked_container.final_refine_single_epoch_result_masks(curr_active_pipeline=owning_pipeline_reference, decoding_time_bin_size=time_bin_size, an_epoch_name=an_epoch_name)
        active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container.final_refine_single_epoch_result_masks(curr_active_pipeline=curr_active_pipeline, fine_decoding_t_bin_size=fine_decoding_time_bin_size, a_decoder_name=an_epoch_name)
        masked_container.debug_computed_dict[an_epoch_name].update({'active_epochs_result': active_epochs_result, 'custom_results_df_list': custom_results_df_list, 'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj})
    except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:
        print(f'\t\tWARN: the `enable_filter_and_final_result_processing` part of `perform_predictive_decoding_analysis(...) failed with error: {e}. Skipping.')
        raise # pass
    
    except Exception as e:
        raise
## END for an_epoch_name in epoch_names...

# step: 0.0004381917184223964
```

# Output
```
trying `.masked_container.final_refine_single_epoch_result_masks(...)` for an_epoch_name: "roam"...
Running in parallel: max_workers=4, use_parallel=True
	finished with for epochs loops doing locality recomputations
step: 0.0243439843567998
	finished with PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(...)
	finished with promenence_result_obj.compute_discrete_contour_masks(...)
about to iterate n_total_epochs: 74 epochs.
Processing 74 epochs in parallel (max_workers=4)...
	row[0/74]	row[1/74]

	row[2/74]	row[3/74]

performing .recompute_all() for epoch....performing .recompute_all() for epoch....
performing .recompute_all() for epoch....

performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[4/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[5/74]
performing .recompute_all() for epoch....
	row[6/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[7/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[8/74]
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[9/74]
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[10/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[11/74]
performing .recompute_all() for epoch....
	row[12/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[13/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[14/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[15/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[16/74]
performing .recompute_all() for epoch....
	row[17/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[18/74]
performing .recompute_all() for epoch....
	row[19/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[20/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[21/74]
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[22/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[23/74]
performing .recompute_all() for epoch....
	row[24/74]
performing .recompute_all() for epoch....
	row[25/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[26/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[27/74]
performing .recompute_all() for epoch....
	row[28/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[29/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[30/74]
performing .recompute_all() for epoch....
	row[31/74]
performing .recompute_all() for epoch....
	row[32/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[33/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[34/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[35/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[36/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[37/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[38/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[39/74]
performing .recompute_all() for epoch....
	row[40/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[41/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[42/74]
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[43/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[44/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[45/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[46/74]
performing .recompute_all() for epoch....
	row[47/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[48/74]
performing .recompute_all() for epoch....
	row[49/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[50/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[51/74]
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[52/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[53/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[54/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[55/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[56/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[57/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[58/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[59/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[60/74]
performing .recompute_all() for epoch....
	row[61/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[62/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[63/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[64/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[65/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[66/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[67/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[68/74]
performing .recompute_all() for epoch....
	row[69/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[70/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[71/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[72/74]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[73/74]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	finished with PredictiveDecoding.compute_specific_future_and_past_analysis(...)
	assigning the results to self.debug_computed_dict...
		done assigning. end of function.
	trying `.masked_container.final_refine_single_epoch_result_masks(...)` for an_epoch_name: "sprinkle"...
Running in parallel: max_workers=4, use_parallel=True
	finished with for epochs loops doing locality recomputations
step: 0.0384608697607823
	finished with PeakPromenence._perform_find_posterior_peaks_peak_prominence2d_computation(...)
	finished with promenence_result_obj.compute_discrete_contour_masks(...)
about to iterate n_total_epochs: 65 epochs.
Processing 65 epochs in parallel (max_workers=4)...
	row[0/65]	row[1/65]
	row[2/65]
	row[3/65]

performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[4/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[5/65]
performing .recompute_all() for epoch....
	row[6/65]
performing .recompute_all() for epoch....
	row[7/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[8/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[9/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[10/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[11/65]
performing .recompute_all() for epoch....
	row[12/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[13/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[14/65]
	row[15/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[16/65]
	row[17/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[18/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[19/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[20/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[21/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[22/65]
performing .recompute_all() for epoch....
	row[23/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
warn: empty df!
	row[24/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[25/65]
performing .recompute_all() for epoch....
	row[26/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[27/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[28/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[29/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[30/65]
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[31/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[32/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[33/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[34/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[35/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[36/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[37/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[38/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[39/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[40/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
	row[41/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[42/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[43/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[44/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[45/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[46/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[47/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[48/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[49/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[50/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[51/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[52/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[53/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[54/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[55/65]
performing .recompute_all() for epoch....
	row[56/65]
performing .recompute_all() for epoch....
	row[57/65]
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[58/65]
	row[59/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[60/65]
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
performing .recompute_all() for epoch....
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\position.py:634: RuntimeWarning: All-NaN axis encountered
  if np.nanmax(np.abs(pos_df['Vp'])) > (2 * np.pi + 1):
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\nanops.py:449: ComplexWarning: Casting complex values to real discards the imaginary part
  values = values.astype("float64")
	row[61/65]
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[62/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
	row[63/65]
performing .recompute_all() for epoch....
performing .recompute_all() for epoch....
	row[64/65]
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
performing .recompute_all() for epoch....
WARNING: Intervals in start_stop_times_arr are normally non-overlapping, but we can continue since we are using the slower determine_unsorted_event_interval_identity(...). Continuing.
		WARN: the `enable_filter_and_final_result_processing` part of `perform_predictive_decoding_analysis(...) failed with error: "None of [Index(['start', 'stop'], dtype='object')] are in the [columns]". Skipping.
```



# TRACEBACK
```python
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
Cell In[19], line 11
      8         masked_container.debug_computed_dict[an_epoch_name] = {}
     10     # active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = a_masked_container.final_refine_single_epoch_result_masks(curr_active_pipeline=owning_pipeline_reference, decoding_time_bin_size=time_bin_size, an_epoch_name=an_epoch_name)
---> 11     active_epochs_result, custom_results_df_list, decoded_epoch_t_bins_promenence_result_obj = masked_container.final_refine_single_epoch_result_masks(curr_active_pipeline=curr_active_pipeline, fine_decoding_t_bin_size=fine_decoding_time_bin_size, a_decoder_name=an_epoch_name)
     12     masked_container.debug_computed_dict[an_epoch_name].update({'active_epochs_result': active_epochs_result, 'custom_results_df_list': custom_results_df_list, 'decoded_epoch_t_bins_promenence_result_obj': decoded_epoch_t_bins_promenence_result_obj})
     13 except (ValueError, AttributeError, IndexError, KeyError, TypeError) as e:

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:3942, in PredictiveDecodingComputationsContainer.final_refine_single_epoch_result_masks(self, curr_active_pipeline, fine_decoding_t_bin_size, a_decoder_name, **kwargs)
   3940 measured_positions_df: pd.DataFrame = deepcopy(self.decoding_locality.measured_positions_df)
   3941 #TODO 2026-01-21 08:45: - [ ] `epoch_t_bin_high_prob_masks_dict ` or `epoch_high_prob_masks_dict` are used to update the final masks
-> 3942 epoch_matching_past_future_positions, _an_out_tuple, a_decoded_filter_epochs_df = PredictiveDecoding.compute_specific_future_and_past_analysis(decoded_local_epochs_result=a_masked_result,
   3943         measured_positions_df=measured_positions_df, gaussian_volume=self.predictive_decoding.gaussian_volume, ## the volume for all time bins,
   3944         active_epochs_df=a_decoded_filter_epochs_df,
   3945         an_epoch_name=a_decoder_name, top_v_percent=None,
   3946         epoch_t_bin_high_prob_masks_dict=epoch_prom_t_bin_high_prob_pos_masks_dict, ## These optional kwargs (epoch_prom_high_prob_pos_masks_dict, epoch_prom_t_bin_high_prob_pos_masks_dict) being set are why the promenece results are actually used this time!
   3947         epoch_high_prob_masks_dict=epoch_prom_high_prob_pos_masks_dict,
   3948         a_slice_multiplier=slice_level_multipliers[0],
   3949         progress_print=True,
   3950         merging_adjacent_max_separation_sec = 1e-9,
   3951         minimum_epoch_duration = 0.05,
   3952         # merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
   3953         should_defer_extended_computations=True, max_workers=max_workers, use_parallel=use_parallel,
   3954 )
   3955 epoch_high_prob_pos_masks, epoch_t_bins_high_prob_pos_masks, epoch_matching_positions, past_future_info_dict, matching_pos_dfs_list, matching_pos_epochs_dfs_list, _out_processed_items_list_dict = _an_out_tuple
   3956 # _out_epoch_flat_mask_future_past_result: List[MatchingPastFuturePositionsResult] = _out_processed_items_list_dict['_out_epoch_flat_mask_future_past_result']
   3957 
   3958 ## OUTPUTS: _out_epoch_flat_mask_future_past_result

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:3048, in PredictiveDecoding.compute_specific_future_and_past_analysis(cls, decoded_local_epochs_result, measured_positions_df, gaussian_volume, active_epochs_df, an_epoch_name, top_v_percent, epoch_t_bin_high_prob_masks_dict, epoch_high_prob_masks_dict, a_slice_multiplier, merging_adjacent_max_separation_sec, minimum_epoch_duration, progress_print, use_parallel, max_workers, should_defer_extended_computations, disable_segmentation, **kwargs)
   3045         futures.append(future)
   3047     for future in as_completed(futures):
-> 3048         results_list.append(future.result())
   3050 # Sort results by index to maintain order
   3051 results_list.sort(key=lambda x: x[0])

File ~\AppData\Roaming\uv\python\cpython-3.9.13-windows-x86_64-none\lib\concurrent\futures\_base.py:439, in Future.result(self, timeout)
    437     raise CancelledError()
    438 elif self._state == FINISHED:
--> 439     return self.__get_result()
    441 self._condition.wait(timeout)
    443 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:

File ~\AppData\Roaming\uv\python\cpython-3.9.13-windows-x86_64-none\lib\concurrent\futures\_base.py:391, in Future.__get_result(self)
    389 if self._exception:
    390     try:
--> 391         raise self._exception
    392     finally:
    393         # Break a reference cycle with the exception in self._exception
    394         self = None

File ~\AppData\Roaming\uv\python\cpython-3.9.13-windows-x86_64-none\lib\concurrent\futures\thread.py:58, in _WorkItem.run(self)
     55     return
     57 try:
---> 58     result = self.fn(*self.args, **self.kwargs)
     59 except BaseException as exc:
     60     self.future.set_exception(exc)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:2862, in PredictiveDecoding._process_single_epoch_future_past_analysis(cls, i, curr_epoch_p_x_given_n, curr_epoch_time_bin_centers, measured_positions_df, top_v_percent, epoch_t_bin_high_prob_masks_dict, epoch_high_prob_masks_dict, a_slice_multiplier, n_epoch_time_bins, merging_adjacent_max_separation_sec, minimum_epoch_duration, progress_print, n_total_epochs, decoded_epoch_result, **kwargs)
   2860 merged_epoch_mask_result = None
   2861 if merged_epoch_mask is not None:
-> 2862     merged_epoch_mask_result: MatchingPastFuturePositionsResult = PredictiveDecoding.detect_matching_past_future_positions(epoch_high_prob_mask=merged_epoch_mask, epoch_t_bins_high_prob_pos_mask=is_high_prob_mask, measured_positions_df=measured_positions_df, curr_epoch_start_t=curr_epoch_start_t, curr_epoch_stop_t=curr_epoch_stop_t, merging_adjacent_max_separation_sec=merging_adjacent_max_separation_sec, minimum_epoch_duration=minimum_epoch_duration,
   2863                                                                                          decoded_epoch_result=decoded_epoch_result, **kwargs)
   2866 return (i, is_high_prob_mask, any_t_Bin_high_prob_pos_mask, any_t_bin_result, merged_epoch_mask, processed_masks, merged_epoch_mask_result)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:2774, in PredictiveDecoding.detect_matching_past_future_positions(cls, epoch_high_prob_mask, measured_positions_df, curr_epoch_start_t, curr_epoch_stop_t, merging_adjacent_max_separation_sec, minimum_epoch_duration, epoch_t_bins_high_prob_pos_mask, decoded_epoch_result, should_defer_extended_computations, **kwargs)
   2771 print(f'performing .recompute_all() for epoch....')        
   2772 # num_pre_found_epochs: int = len(_out_obj.matching_pos_epochs_df)
   2773 # if (num_pre_found_epochs < 3): # def recompute, it's always 2 epochs when it's wrong
-> 2774 _out_obj.recompute_all() ## almost 30.0 seconds just for one epoch
   2775 # num_post_found_epochs: int = len(_out_obj.matching_pos_epochs_df)
   2776 # print(f"num_post_found_epochs: {num_post_found_epochs}")        
   2777 
   (...)
   2789 #     epoch_only_relevant_positions_df, pos_segment_to_centroid_seq_segment_idx_map = PosteriorMaskPostProcessing._compare_centroid_and_pos_traj_angle(a_pos_df=epoch_only_relevant_positions_df, 
   2790 #                                                                                                                                                      a_centroids_search_segments_df=_out_obj.a_centroids_search_segments_df)
   2793 return _out_obj

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:1515, in MatchingPastFuturePositionsResult.recompute_all(self)
   1513 def recompute_all(self):
   1514     """ performs all recomputations like it would at init if `self.should_defer_extended_computations` were not False """    
-> 1515     self._recompute_all_pos_dfs()
   1516     if (self.epoch_t_bins_high_prob_pos_mask) is not None and (self.decoded_epoch_result is not None):
   1517         self._recompute_high_prob_mask_centroids()

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:1577, in MatchingPastFuturePositionsResult._recompute_all_pos_dfs(self)
   1574             self.relevant_positions_df.loc[epoch_pos_df.index, col] = epoch_pos_df[col]
   1576 ## re-index:
-> 1577 self.relevant_positions_df = self._recompute_relevant_pos_epoch_position_df_index_column(a_matching_pos_epochs_df=self.matching_pos_epochs_df, relevant_positions_df=self.relevant_positions_df, drop_non_epoch_events=False, epoch_id_key_name=self.epoch_id_key_name) ## drop those that aren't in the epochs
   1579 #TODO 2026-01-23 12:32: - [ ] Recompute the complete path
   1580 ## I think the compute order is correct, but can't be sure
   1581 
   1582 ## OUTPUTS: matching_relevant_positions_df
   1583 merged_segment_epochs, relevant_merged_positions_df, matching_pos_epochs_df = self.compute_compilete_paths(max_allowed_trajectory_gap_seconds=self.max_allowed_trajectory_gap_seconds, merged_found_pos_epoch_id_key_name=self.merged_found_pos_epoch_id_key_name)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py:1636, in MatchingPastFuturePositionsResult._recompute_relevant_pos_epoch_position_df_index_column(cls, a_matching_pos_epochs_df, relevant_positions_df, epoch_id_key_name, drop_non_epoch_events)
   1633 else:
   1634     a_matching_pos_epochs_df['label'] = a_matching_pos_epochs_df['label'].astype(int)
-> 1636 relevant_positions_df = relevant_positions_df.time_point_event.adding_epochs_identity_column(epochs_df=a_matching_pos_epochs_df, epoch_id_key_name=epoch_id_key_name, override_time_variable_name='t',
   1637                                                     # epoch_label_column_name='label', no_interval_fill_value=np.nan,
   1638                                                     epoch_label_column_name='label', no_interval_fill_value=-1,
   1639                                                     should_replace_existing_column=True, drop_non_epoch_events=drop_non_epoch_events,
   1640                                                     overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH) ## #TODO 2026-01-15 06:13: - [ ] KeyError: "None of [Index(['start', 'stop'], dtype='object')] are in the [columns]"
   1643 # TODO:_custom_build_sequential_position_epochs
   1645 return relevant_positions_df

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\time_slicing.py:267, in TimePointEventAccessor.adding_epochs_identity_column(self, epochs_df, epoch_id_key_name, epoch_label_column_name, override_time_variable_name, no_interval_fill_value, should_replace_existing_column, drop_non_epoch_events, overlap_behavior)
    264     override_time_variable_name = self.time_variable_name # 't_rel_seconds'
    266 self._obj[epoch_id_key_name] = no_interval_fill_value # initialize the column to -1
--> 267 self._obj = add_epochs_id_identity(self._obj, epochs_df=epochs_df, epoch_id_key_name=epoch_id_key_name, epoch_label_column_name=epoch_label_column_name, no_interval_fill_value=no_interval_fill_value, override_time_variable_name=override_time_variable_name, overlap_behavior=overlap_behavior) # uses new add_epochs_id_identity method which is general
    268 if drop_non_epoch_events:
    269     active_point_events_df = self._obj.copy()

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\time_slicing.py:543, in add_epochs_id_identity(spk_df, epochs_df, epoch_id_key_name, epoch_label_column_name, override_time_variable_name, no_interval_fill_value, overlap_behavior)
    520 def add_epochs_id_identity(spk_df, epochs_df, epoch_id_key_name='temp_epoch_id', epoch_label_column_name='label', override_time_variable_name=None, no_interval_fill_value=np.nan, overlap_behavior=OverlappingIntervalsFallbackBehavior.FALLBACK_TO_SLOW_SEARCH):
    521     """ Adds the epoch IDs to each spike in spikes_df as a column named epoch_id_key_name
    522     
    523     NOTE: you can use this for non-spikes dataframes by providing `override_time_variable_name='t'`
   (...)
    540         spike_pbe_identity_arr # Elapsed Time (seconds) = 90.92654037475586, 93.46184754371643, 90.16610431671143 , 89.04321789741516
    541     """
--> 543     spike_epoch_identity_arr = _compute_spike_arbitrary_provided_epoch_ids(spk_df, epochs_df, epoch_label_column_name=epoch_label_column_name, override_time_variable_name=override_time_variable_name, no_interval_fill_value=no_interval_fill_value, overlap_behavior=overlap_behavior)
    544     spk_df[epoch_id_key_name] = spike_epoch_identity_arr
    545     return spk_df

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\mixins\time_slicing.py:450, in _compute_spike_arbitrary_provided_epoch_ids(spk_df, provided_epochs_df, epoch_label_column_name, no_interval_fill_value, override_time_variable_name, overlap_behavior, debug_print)
    448 active_time_variable_name: str = (override_time_variable_name or spk_df.spikes.time_variable_name) # by default use spk_df.spikes.time_variable_name, but an optional override can be provided (to ensure compatibility with PBEs)
    449 spk_times_arr = spk_df[active_time_variable_name].to_numpy()
--> 450 curr_epochs_start_stop_arr = provided_epochs_df[['start','stop']].to_numpy()
    451 if epoch_label_column_name is None:
    452     curr_epoch_identity_labels = provided_epochs_df.index.to_numpy() # currently using the index instead of the label.

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\frame.py:3813, in DataFrame.__getitem__(self, key)
   3811     if is_iterator(key):
   3812         key = list(key)
-> 3813     indexer = self.columns._get_indexer_strict(key, "columns")[1]
   3815 # take() does not accept boolean indexers
   3816 if getattr(indexer, "dtype", None) == bool:

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\indexes\base.py:6070, in Index._get_indexer_strict(self, key, axis_name)
   6067 else:
   6068     keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
-> 6070 self._raise_if_missing(keyarr, indexer, axis_name)
   6072 keyarr = self.take(indexer)
   6073 if isinstance(key, Index):
   6074     # GH 42790 - Preserve name from an Index

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\pandas\core\indexes\base.py:6130, in Index._raise_if_missing(self, key, indexer, axis_name)
   6128     if use_interval_msg:
   6129         key = list(key)
-> 6130     raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6132 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
   6133 raise KeyError(f"{not_found} not in index")

KeyError: "None of [Index(['start', 'stop'], dtype='object')] are in the [columns]"
```