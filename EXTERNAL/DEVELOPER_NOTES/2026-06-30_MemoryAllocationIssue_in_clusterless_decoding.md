```python
hardcoded_params.decoder_building_session_names: ['maze1', 'maze2', 'maze_GLOBAL']
hardcoded_params.non_global_activity_session_names: ['maze1', 'maze2']
active_data_session_types_registered_classes_dict: {'bapun': <class 'neuropy.core.session.Formats.Specific.BapunDataSessionFormat.BapunDataSessionFormatRegisteredClass'>, 'kdiba': <class 'neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat.KDibaOldDataSessionFormatRegisteredClass'>, 'rachel': <class 'neuropy.core.session.Formats.Specific.RachelDataSessionFormat.RachelDataSessionFormat'>}
fixing up session computation epochs...
	done. new epochs: 
6 epochs
array([[0, 11400],
       [11510, 14693],
       [11510, 22860],
       [14820, 19140],
       [19200, 22860],
       [22980, 54130]])


curr_epoch_names: ['pre', 'maze1', 'maze_GLOBAL', 'post1', 'maze2', 'post2']
computing linearized position for session using method="shapely"...
estimating the laps from the linear position...
estimating the maze_id to laps...
filtering sessions via `curr_active_pipeline.filter_sessions(...)`...
Applying session filter named "maze1"...
WARNING: SpikesAccessor.set_time_variable_name(new_time_variable_name: t_seconds) has been called. Be careful!
	 no change in time_variable_name. It will remain t_seconds.
Applying session filter named "maze_GLOBAL"...
WARNING: SpikesAccessor.set_time_variable_name(new_time_variable_name: t_seconds) has been called. Be careful!
	 no change in time_variable_name. It will remain t_seconds.
Applying session filter named "maze2"...
WARNING: SpikesAccessor.set_time_variable_name(new_time_variable_name: t_seconds) has been called. Be careful!
	 no change in time_variable_name. It will remain t_seconds.
	hardcoded_params.grid_bin_bounds: ((-435.0, 320.0), (-508.0, 430.0))
beginning compute...
i: 0, active_epoch_names: ['maze1', 'maze2']
Performing perform_action_for_all_contexts with action EvaluationActions.EVALUATE_COMPUTATIONS on filtered_session with filter named "maze1"...
curr_active_computation_params.pf_params.computation_epochs: 1 epochs
array([[11510, 14693]])

due to includelist, including only 2 out of 17 registered computation functions.
Performing _execute_computation_functions(...) with 2 registered_computation_functions...
Recomputing active_epoch_placefields1D... 	 done.
Recomputing active_epoch_placefields2D... 	 done.
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\continuous_state_transitions.py:26: RuntimeWarning: invalid value encountered in divide
  x /= x.sum(axis=1, keepdims=True)
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\likelihoods\multiunit_likelihood.py:101: RuntimeWarning: divide by zero encountered in log
  return np.log(mean_rate) + np.log(density) - np.log(occupancy)
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\likelihoods\multiunit_likelihood.py:101: RuntimeWarning: invalid value encountered in subtract
  return np.log(mean_rate) + np.log(density) - np.log(occupancy)
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\core.py:210: NumbaPerformanceWarning: '@' is faster on contiguous arrays, called on (Array(float64, 2, 'F', False, aligned=True), Array(float64, 2, 'A', False, aligned=True))
  discrete_state_transition[state_k, state_k_1]
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\continuous_state_transitions.py:26: RuntimeWarning: invalid value encountered in divide
  x /= x.sum(axis=1, keepdims=True)
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\likelihoods\multiunit_likelihood.py:101: RuntimeWarning: divide by zero encountered in log
  return np.log(mean_rate) + np.log(density) - np.log(occupancy)
h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\likelihoods\multiunit_likelihood.py:101: RuntimeWarning: invalid value encountered in subtract
  return np.log(mean_rate) + np.log(density) - np.log(occupancy)

---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[6], line 7
      5 time_bin_size: float = 0.050 # 50ms bins
      6 active_computation_functions_name_includelist = ['pf_computation', 'position_decoding_clusterless'] # , 'pfdt_computation'
----> 7 curr_active_pipeline = final_process_bapun_all_comps(curr_active_pipeline=curr_active_pipeline, active_data_mode_name='bapun',
      8                                                     posthoc_save=False,
      9                                                     time_bin_size=time_bin_size,
     10                                                     # time_bin_size=0.250,
     11                                                     # overwrite_extant = False,
     12                                                     overwrite_extant = True,
     13                                                     # fail_on_exception = False,
     14                                                     fail_on_exception=True,
     15                                                     active_computation_functions_name_includelist=active_computation_functions_name_includelist,
     16 
     17 )
     18 # except Exception as e:
     19 #     print(f'exception: {e}')
     20 #     # raise e
     21 #     pass    
     22 
     23 ## 9m

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py:5223, in final_process_bapun_all_comps(curr_active_pipeline, posthoc_save, override_parameters_flat_keypaths_dict, active_data_mode_name, time_bin_size, overwrite_extant, **kwargs)
   5216 def final_process_bapun_all_comps(curr_active_pipeline, posthoc_save: bool=True, override_parameters_flat_keypaths_dict=None, active_data_mode_name = 'bapun', time_bin_size=0.5, overwrite_extant: bool=False, **kwargs):
   5217     """ Main non-kdiba processing/computation function (for Bapun/Rachel/DANDI NWB/etc sessions)
   5218     
   5219     from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import final_process_bapun_all_comps
   5220     curr_active_pipeline = final_process_bapun_all_comps(curr_active_pipeline=curr_active_pipeline, posthoc_save=True)
   5221     
   5222     """
-> 5223     return final_process_non_kdiba_all_comps(curr_active_pipeline, active_data_mode_name=active_data_mode_name, posthoc_save=posthoc_save, override_parameters_flat_keypaths_dict=override_parameters_flat_keypaths_dict, time_bin_size=time_bin_size, overwrite_extant=overwrite_extant, **kwargs)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py:5504, in final_process_non_kdiba_all_comps(curr_active_pipeline, active_data_mode_name, posthoc_save, override_parameters_flat_keypaths_dict, time_bin_size, overwrite_extant, **kwargs)
   5499     print(f'i: {i}, active_epoch_names: {active_epoch_names}') # (activity_only_epoch_names)
   5501     #BUG 2026-06-23 08:35: - [ ] IMPORTANT: `overwrite_extant_results` should never be True in the following call, or else each run overwrites all previous ones so you only end up with the last filtered session:
   5502     # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_excludelist=['_perform_spike_burst_detection_computation', '_perform_velocity_vs_pf_density_computation', '_perform_velocity_vs_pf_simplified_count_density_computation']) # SpikeAnalysisComputations._perform_spike_burst_detection_computation
   5503     # curr_active_pipeline.perform_computations(active_session_computation_configs[0], computation_functions_name_includelist=active_computation_functions_name_includelist, enabled_filter_names=activity_only_epoch_names, overwrite_extant_results=True, fail_on_exception=False, debug_print=True) # SpikeAnalysisComputations._perform_spike_burst_detection_computation
-> 5504     curr_active_pipeline.perform_computations(a_config, computation_functions_name_includelist=active_computation_functions_name_includelist, enabled_filter_names=active_epoch_names, overwrite_extant_results=False, fail_on_exception=kwargs.get('fail_on_exception', False), debug_print=True) # SpikeAnalysisComputations._perform_spike_burst_detection_computation
   5505     # curr_active_pipeline.perform_computations(a_config, computation_functions_name_includelist=active_computation_functions_name_includelist, enabled_filter_names=active_epoch_names, overwrite_extant_results=False, fail_on_exception=False, debug_print=True)
   5506 
   5507 # ==================================================================================================================================================================================================================================================================================== #
   5508 # COMPUTE DONE                                                                                                                                                                                                                                                                         #
   5509 # ==================================================================================================================================================================================================================================================================================== #
   5510 print(f'\tcompute done!')

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:2347, in PipelineWithComputedPipelineStageMixin.perform_computations(self, active_computation_params, enabled_filter_names, overwrite_extant_results, computation_functions_name_includelist, computation_functions_name_excludelist, fail_on_exception, debug_print)
   2342 progress_logger_callback=(lambda x: self.logger.info(x))
   2343 # self.stage.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
   2344 #     computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)
   2345 
   2346 # Calls self.stage's version:
-> 2347 self.stage.perform_computations(enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
   2348     computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)
   2350 # Global MultiContext computations will be done here:
   2351 if progress_logger_callback is not None:

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1301, in ComputedPipelineStage.perform_computations(self, active_computation_params, enabled_filter_names, overwrite_extant_results, computation_functions_name_includelist, computation_functions_name_excludelist, fail_on_exception, debug_print, progress_logger_callback)
   1280 """The main computation function for the pipeline.
   1281 
   1282 Wraps `perform_action_for_all_contexts`
   (...)
   1298     factored out of `NeuropyPipeline` for use in GlobalComputationFunctions
   1299 """
   1300 assert (self.can_compute), "Current stage must already be a ComputedPipelineStage. Call self.filter_sessions with filter configs to reach this step."
-> 1301 self.perform_action_for_all_contexts(EvaluationActions.EVALUATE_COMPUTATIONS, enabled_filter_names=enabled_filter_names, active_computation_params=active_computation_params, overwrite_extant_results=overwrite_extant_results,
   1302     computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, debug_print=debug_print)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1005, in ComputedPipelineStage.perform_action_for_all_contexts(self, action, enabled_filter_names, active_computation_params, overwrite_extant_results, computation_functions_name_includelist, computation_functions_name_excludelist, fail_on_exception, progress_logger_callback, are_global, debug_print)
   1001             skip_computations_for_this_result = True
   1003     if not skip_computations_for_this_result:
   1004         # call to perform any registered computations:
-> 1005         active_computation_results[a_select_config_name] = self.perform_registered_computations_single_context(active_computation_results[a_select_config_name],
   1006             computation_functions_name_includelist=computation_functions_name_includelist, computation_functions_name_excludelist=computation_functions_name_excludelist, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)
   1008 elif action.name == EvaluationActions.RUN_SPECIFIC.name:
   1009     print(f'Performing run_specific_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:492, in ComputedPipelineStage.perform_registered_computations_single_context(self, previous_computation_result, computation_functions_name_includelist, computation_functions_name_excludelist, fail_on_exception, progress_logger_callback, are_global, debug_print)
    488         print(f'due to excludelist, including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions.')
    489         # TODO: do something about the previous_computation_result?
    490 
    491 # Perform the computations:
--> 492 return ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result=previous_computation_result, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1418, in ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
   1416 if progress_logger_callback is not None:
   1417     progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
-> 1418 previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here ## #TODO 2025-02-19 13:51: - [ ] was getting`TypeError: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation() argument after ** must be a mapping, not NoneType` which I fixed by replacing any None in the list with {}
   1419 # Log the computation copmlete time:
   1420 computation_times[computation_times_key_fn(f)] = datetime.now()

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\DefaultComputationFunctions.py:107, in DefaultComputationFunctions._perform_clusterless_position_decoding_computation(computation_result, sampling_frequency_hz, multiunits, rtc_time, clusterless_params, **kwargs)
    105 computation_result.computed_data['pf1D_ClusterlessDecoder'] = _build_decoder_for_pf(computation_result.computed_data['pf1D'])
    106 if ('pf2D' in computation_result.computed_data) and (computation_result.computed_data.get('pf2D', None) is not None):
--> 107     computation_result.computed_data['pf2D_ClusterlessDecoder'] = _build_decoder_for_pf(computation_result.computed_data['pf2D'])
    108 else:
    109     computation_result.computed_data['pf2D_ClusterlessDecoder'] = None

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\DefaultComputationFunctions.py:102, in DefaultComputationFunctions._perform_clusterless_position_decoding_computation.<locals>._build_decoder_for_pf(pf)
    100     pf_multiunits, pf_rtc_time = build_multiunits_from_session(sess, clusterless_params.clusterless_sampling_frequency_hz, t_start, t_end, spikes_df=pf.filtered_spikes_df.copy())
    101 decoder = ClusterlessRTCPositionDecoder(pf=pf, sampling_frequency_hz=clusterless_params.clusterless_sampling_frequency_hz, multiunits=pf_multiunits, rtc_time=pf_rtc_time, clusterless_params=clusterless_params, setup_on_init=True, post_load_on_init=False, debug_print=False)
--> 102 decoder.compute_all()
    103 return decoder

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\rtc_clusterless_decoder.py:86, in ClusterlessRTCPositionDecoder.compute_all(self, debug_print)
     84 self.is_training_mask = is_training
     85 self.classifier.fit(position_train, multiunits_train, is_training=is_training)
---> 86 self.rtc_results = self.classifier.predict(multiunits_train, time=self.rtc_time[:len(multiunits_train)], is_compute_acausal=True, use_gpu=False)
     87 self.p_x_given_n = rtc_posterior_to_p_x_given_n(self.rtc_results, self.pf, state_index=params.state_index_for_posterior)
     88 self.flat_p_x_given_n = self.p_x_given_n.reshape(self.flat_position_size, self.num_time_windows) if self.p_x_given_n.ndim > 2 else self.p_x_given_n

File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\classifier.py:1335, in ClusterlessClassifier.predict(self, multiunits, time, is_compute_acausal, use_gpu, state_names, store_likelihood)
   1331     is_track_interior = self.environments[env_ind].is_track_interior_.ravel(
   1332         order="F"
   1333     )
   1334     place_bin_centers = self.environments[env_ind].place_bin_centers_
-> 1335     likelihood[(env_name, enc_group)] = _ClUSTERLESS_ALGORITHMS[
   1336         self.clusterless_algorithm
   1337     ][1](
   1338         multiunits=multiunits,
   1339         place_bin_centers=place_bin_centers,
   1340         is_track_interior=is_track_interior,
   1341         **encoding_params,
   1342     )
   1343 if store_likelihood:
   1344     self.likelihood_ = likelihood

File h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\replay_trajectory_classification\likelihoods\multiunit_likelihood.py:410, in estimate_multiunit_likelihood(multiunits, encoding_marks, mark_std, place_bin_centers, encoding_positions, position_std, occupancy, mean_rates, summed_ground_process_intensity, bin_diffusion_distances, edges, max_mark_diff, set_diag_zero, is_track_interior, time_bin_size, block_size, ignore_no_spike, disable_progress_bar, use_diffusion)
    403     log_likelihood = (
    404         -time_bin_size
    405         * summed_ground_process_intensity
    406         * np.zeros((n_time, 1), dtype=np.float32)
    407     )
    408 else:
    409     log_likelihood = (
--> 410         -time_bin_size
    411         * summed_ground_process_intensity
    412         * np.ones((n_time, 1), dtype=np.float32)
    413     )
    415 multiunits = np.moveaxis(multiunits, -1, 0)
    416 n_position_bins = is_track_interior.sum()

MemoryError: Unable to allocate 536. GiB for an array with shape (3179092, 45267) and data type float32
```