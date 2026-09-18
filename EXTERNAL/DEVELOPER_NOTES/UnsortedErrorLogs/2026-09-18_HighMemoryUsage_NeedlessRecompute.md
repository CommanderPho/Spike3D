```CURR_BATCH_OUTPUT_PREFIX: 2026-09-18_Apogee-2006-6-09_1-22-43
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(curr_session_context: kdiba_gor01_one_2006-6-09_1-22-43_normal_computed_[1, 2, 4, 6, 7, 9]_5.0, curr_session_basedir: W:\Data\KDIBA\gor01\one\2006-6-09_1-22-43, ...)
	test_display_output_path: "K:\scratch\collected_outputs\kdiba_gor01_one_2006-6-09_1-22-43"
	 trying "_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay"
	computing required decoded results at time_bin_size: 0.02 before plotting...
	ordered_required_dependent_computation_fn_names: ['_split_to_directional_laps', 'perform_compute_non_PBE_epochs', '_build_merged_directional_placefields', 'perform_generalized_specific_epochs_decoding', '_decode_continuous_using_directional_decoders']
included includelist is specified: ['_split_to_directional_laps', 'perform_compute_non_PBE_epochs', '_build_merged_directional_placefields', 'perform_generalized_specific_epochs_decoding', '_decode_continuous_using_directional_decoders'], so only performing these extended computations.
Running batch_evaluate_required_computations(...) with global_epoch_name: "maze_any"
done with all batch_evaluate_required_computations(...).
	have 5 functions to compute: ['_split_to_directional_laps', 'perform_compute_non_PBE_epochs', '_build_merged_directional_placefields', 'perform_generalized_specific_epochs_decoding', '_decode_continuous_using_directional_decoders']. Performing specific computations: ....
for global computations: Performing run_specific_computations_single_context(..., computation_functions_name_includelist=['_split_to_directional_laps', 'perform_compute_non_PBE_epochs', '_build_merged_directional_placefields', 'perform_generalized_specific_epochs_decoding', '_decode_continuous_using_directional_decoders'], ...)...
	run_specific_computations_single_context(including only 5 out of 18 registered computation functions): active_computation_functions: [<function DirectionalPlacefieldGlobalComputationFunctions._split_to_directional_laps at 0x00000245E0695940>, <function DirectionalPlacefieldGlobalComputationFunctions._build_merged_directional_placefields at 0x00000245E06959D0>, <function DirectionalPlacefieldGlobalComputationFunctions._decode_continuous_using_directional_decoders at 0x00000245E0695A60>, <function EpochComputationFunctions.perform_compute_non_PBE_epochs at 0x0000024843FA9670>, <function EpochComputationFunctions.perform_generalized_specific_epochs_decoding at 0x0000024843FA9160>]...
Performing _execute_computation_functions(...) with 5 registered_computation_functions...
Performing _execute_computation_functions(...) with 5 registered_computation_functions...
Executing [0/5]: <function DirectionalPlacefieldGlobalComputationFunctions._split_to_directional_laps at 0x000002476D024B80>
WARN: _split_to_directional_laps(...): include_includelist: ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any'] is specified but include_includelist is currently ignored! Continuing with defaults.
DirectionalLapsResult.init_from_pipeline_natural_epochs(...): was_modified: False
Executing [1/5]: <function DirectionalPlacefieldGlobalComputationFunctions._build_merged_directional_placefields at 0x00000248137498B0>
10{"stdout":"[{\"variableName\": \"ID_TO_MEANING\", \"type\": \"dictionary\", \"supportedEngines\": [\"pandas\"], \"isLocalVariable\": true, \"rawType\": \"builtins.dict\"}, {\"variableName\": \"NULL\", \"type\": \"unknown\", \"supportedEngines\": [\"pandas\"], \"isLocalVariable\": true, \"rawType\": \"_pydevd_bundle.pydevd_constants.Null\"}]\n","stderr":"","mime":[]}
Executing [2/5]: <function DirectionalPlacefieldGlobalComputationFunctions._decode_continuous_using_directional_decoders at 0x0000024813749310>
should_disable_cache == True so setting had_existing_DirectionalDecodersDecoded_result = False
	had_existing_DirectionalDecodersDecoded_result == False. New DirectionalDecodersContinuouslyDecodedResult will be built...
	time_bin_size: 0.02
	 computation done. Creating new DirectionalDecodersContinuouslyDecodedResult....
Executing [3/5]: <function EpochComputationFunctions.perform_compute_non_PBE_epochs at 0x00000248137490D0>
perform_compute_non_PBE_epochs(..., training_data_portion=0.8333333333333334, epochs_decoding_time_bin_size: 0.02, frame_divide_bin_size: 10.0)
available RAM: 16.79 GB
Total memory required: 7.00 GB
WARN: perform_compute_non_PBE_epochs(...): include_includelist: ['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', 'maze2_even', 'maze_even', 'maze1_any', 'maze2_any', 'maze_any'] is specified but include_includelist is currently ignored! Continuing with defaults.
Uses 1D Placefields
epochs_decoding_time_bin_size = 0.02, frame_divide_bin_size = 10.0
WARN: Epoch[45]: with 5 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: Epoch[60]: with 7 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: Epoch[113]: with 16 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: Epoch[168]: with 6 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: Epoch[344]: with 24 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: Epoch[365]: with 4 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: Epoch[379]: with 23 time_bins has no time bins with enough firing to infer back-filled positions from, so all entries will be NaN.
WARN: `_build_output_decoded_posteriors`: computing for a_decoded_epoch_type_name: "non_pbe" failed, likely because all epochs were filtered out. Error: need at least one array to concatenate.
	... enable_fail_on_empty_exception == False so continuing anyway.
Executing [4/5]: <function EpochComputationFunctions.perform_generalized_specific_epochs_decoding at 0x0000024813749A60>
	epochs_decoding_time_bin_size: 0.02
removed previous "EpochComputations.a_generic_decoder_dict_decoded_epochs_dict_result" result and computing fresh since `drop_previous_result_and_compute_fresh == True`
	 dropping "EpochComputations" and recomputing...
for global computations: Performing run_specific_computations_single_context(..., computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring', 'non_PBE_epochs_results'], ...)...
	run_specific_computations_single_context(including only 5 out of 18 registered computation functions): active_computation_functions: [<function DirectionalPlacefieldGlobalComputationFunctions._build_merged_directional_placefields at 0x00000245E06959D0>, <function DirectionalPlacefieldGlobalComputationFunctions._decode_continuous_using_directional_decoders at 0x00000245E0695A60>, <function DirectionalPlacefieldGlobalComputationFunctions._decode_and_evaluate_epochs_using_directional_decoders at 0x00000245E0695AF0>, <function DirectionalPlacefieldGlobalComputationFunctions._decoded_epochs_heuristic_scoring at 0x00000245E0695B80>, <function EpochComputationFunctions.perform_compute_non_PBE_epochs at 0x0000024843FA9670>]...
Performing _execute_computation_functions(...) with 5 registered_computation_functions...
Executing [0/5]: <function DirectionalPlacefieldGlobalComputationFunctions._build_merged_directional_placefields at 0x00000247252535E0>
Executing [1/5]: <function DirectionalPlacefieldGlobalComputationFunctions._decode_continuous_using_directional_decoders at 0x0000024725024D30>
	had_existing_DirectionalDecodersDecoded_result == True. Using existing result and updating.
	time_bin_size: 0.02
(cache_key == (0.02, 0.02)) already found in cache. Not recomputing.
Executing [2/5]: <function DirectionalPlacefieldGlobalComputationFunctions._decode_and_evaluate_epochs_using_directional_decoders at 0x0000024663A66670>
laps_decoding_time_bin_size: 0.02, ripple_decoding_time_bin_size: 0.02, pos_bin_size: 4.877453969028168
laps_decoding_time_bin_size: 0.02, ripple_decoding_time_bin_size: 0.02, pos_bin_size: 4.877453969028168
Performance: WCorr:
	Laps:
agreeing_rows_count/num_total_epochs: 39/82
	agreeing_rows_ratio: 0.47560975609756095
Performance: Ripple: WCorr
agreeing_rows_count/num_total_epochs: 140/416
	agreeing_rows_ratio: 0.33653846153846156
Performance: Simple PF PearsonR:
	Laps:
agreeing_rows_count/num_total_epochs: 21/82
	agreeing_rows_ratio: 0.25609756097560976
Performance: Ripple: Simple PF PearsonR
agreeing_rows_count/num_total_epochs: 103/416
	agreeing_rows_ratio: 0.24759615384615385
Executing [3/5]: <function DirectionalPlacefieldGlobalComputationFunctions._decoded_epochs_heuristic_scoring at 0x0000024663A66940>
same_thresh_cm: 10.700000000000001
10{"stdout":"[{\"variableName\": \"ID_TO_MEANING\", \"type\": \"dictionary\", \"supportedEngines\": [\"pandas\"], \"isLocalVariable\": true, \"rawType\": \"builtins.dict\"}, {\"variableName\": \"NULL\", \"type\": \"unknown\", \"supportedEngines\": [\"pandas\"], \"isLocalVariable\": true, \"rawType\": \"_pydevd_bundle.pydevd_constants.Null\"}]\n","stderr":"","mime":[]}
Executing [4/5]: <function EpochComputationFunctions.perform_compute_non_PBE_epochs at 0x0000024663A66C10>
perform_compute_non_PBE_epochs(..., training_data_portion=0.8333333333333334, epochs_decoding_time_bin_size: 0.02, frame_divide_bin_size: 10.0)
available RAM: 12.66 GB
Total memory required: 14.00 GB
Memory breakdown (GB):
	spike_counts_1D: 0.072
	firing_rates_1D: 0.072
	position_decoded_1D: 0.001
	posterior_1D: 0.041
	occupancy_1D: 0.000
	spike_counts_2D: 0.072
	firing_rates_2D: 0.072
	position_decoded_2D: 0.001
	posterior_2D: 0.047
	occupancy_2D: 0.000
Insufficient memory: Estimated Insufficient Memory: Operation would require 14.00 GB (have 12.66 GB available.
	figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(...): "_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay" failed with error: Estimated Insufficient Memory: Operation would require 14.00 GB (have 12.66 GB available.
 skipping.
Traceback (most recent call last):
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py", line 4741, in figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function
    _out = curr_active_pipeline.display('_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay', display_context, defer_render=True, save_figure=True,
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Display.py", line 633, in display
    curr_display_output = display_function(self, self.global_computation_results, self.computation_results, self.active_configs, active_config_name=None, **kwargs) # CALL GLOBAL DISPLAY FUNCTION
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py", line 2837, in _display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay
    owning_pipeline_reference.resolve_and_execute_full_required_computation_plan(computation_functions_name_includelist=computation_functions_name_includelist,
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 58, in wrapper
    return getattr(self.stage, func.__name__)(*args, **kwargs)
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 1201, in resolve_and_execute_full_required_computation_plan
    self.perform_specific_computation(computation_functions_name_includelist=functions_to_run, computation_kwargs_list=aligned_kwargs_list, enabled_filter_names=enabled_filter_names, fail_on_exception=fail_on_exception, debug_print=debug_print)
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 1308, in perform_specific_computation
    self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 603, in run_specific_computations_single_context
    return ComputedPipelineStage._execute_computation_functions(active_found_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=active_found_computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 1498, in _execute_computation_functions
    previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here ## #TODO 2025-02-19 13:51: - [ ] was getting`TypeError: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation() argument after ** must be a mapping, not NoneType` which I fixed by replacing any None in the list with {}
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\MultiContextComputationFunctions.py", line 19, in _
    x[1] = global_comp_fcn(*x, **kwargs) # update global_computation_results
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py", line 2181, in perform_generalized_specific_epochs_decoding
    a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(curr_active_pipeline=owning_pipeline_reference, force_recompute=force_recompute, time_bin_size=epochs_decoding_time_bin_size, debug_print=debug_print)
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\context_dependent.py", line 1141, in batch_user_compute_fn
    curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring', 'non_PBE_epochs_results'],
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 1308, in perform_specific_computation
    self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 603, in run_specific_computations_single_context
    return ComputedPipelineStage._execute_computation_functions(active_found_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=active_found_computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 1498, in _execute_computation_functions
    previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here ## #TODO 2025-02-19 13:51: - [ ] was getting`TypeError: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation() argument after ** must be a mapping, not NoneType` which I fixed by replacing any None in the list with {}
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\MultiContextComputationFunctions.py", line 19, in _
    x[1] = global_comp_fcn(*x, **kwargs) # update global_computation_results
  File "H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py", line 1994, in perform_compute_non_PBE_epochs
    raise MemoryError(memory_error_msg)
MemoryError: Estimated Insufficient Memory: Operation would require 14.00 GB (have 12.66 GB available.```





```---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[9], line 14
     10 _across_session_results_extended_dict = {}
     12 complete_session_context, (session_context, additional_session_context) = curr_active_pipeline.get_complete_session_context()
---> 14 _across_session_results_extended_dict = _across_session_results_extended_dict | figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(a_dummy, None,
     15                                                     curr_session_context=complete_session_context,
     16                                                     curr_session_basedir=curr_active_pipeline.sess.basepath.resolve(), curr_active_pipeline=curr_active_pipeline,
     17                                                     across_session_results_extended_dict=_across_session_results_extended_dict,
     18                                                     # extreme_threshold=0.5, opacity_max=0.7, thickness_ramping_multiplier=35,
     19                                                     # extreme_threshold=0.8, opacity_max=0.7, thickness_ramping_multiplier=100,
     20                                                     # extreme_threshold=0.5, included_figures_names=['_display_decoded_trackID_marginal_hairy_position'],
     21                                                     included_figures_names=[
     22 														# '_display_plot_decoded_epoch_slices',
     23 														# '_display_directional_merged_pf_decoded_stacked_epoch_slices',
     24                                                         '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay',
     25                                                     ],
     26 													# included_figures_names=['_render_export_all_time_tracks'],
     27                                                     # included_figures_names=['_display_generalized_decoded_yellow_blue_marginal_epochs', '_display_decoded_trackID_marginal_hairy_position', '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay'],
     28                                                     # included_figures_names=['_display_directional_merged_pf_decoded_stacked_epoch_slices', '_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay'],
     29                                                     display_function_kwargs_dict = {'_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay': dict(time_bin_size = 0.02), }, #TODO 2026-09-18 08:58: - [ ] Overrides for Apogee
     30                                                     fail_on_exception_for_debugging=True,
     31                                                 )
     33 # _across_session_results_extended_dict
     34 
     35 # 7m 17s

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py:4741, in figures_plot_generalized_decode_epochs_dict_and_export_results_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict, included_figures_names, display_function_kwargs_dict, extreme_threshold, opacity_max, thickness_ramping_multiplier, fail_on_exception_for_debugging, export_filename_extra_suffix_parts, **additional_marginal_overlaying_measured_position_kwargs)
   4739 a_params_kwargs = dict(time_bin_size = 0.02) | display_function_kwargs_dict.get('_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay', {})
   4740 display_context = curr_active_pipeline.build_display_context_for_session(display_fn_name='trackID_weighted_position_posterior')
-> 4741 _out = curr_active_pipeline.display('_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay', display_context, defer_render=True, save_figure=True,
   4742                                     # override_fig_man=custom_fig_man, 
   4743                                     parent_output_folder=custom_figure_output_path,
   4744                                     **a_params_kwargs,
   4745                                 )
   4747 # _out = EpochComputationDisplayFunctions._display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay(curr_active_pipeline, None, None, None, include_includelist=None, save_figure=True)
   4748 keys_to_convert_to_benedict = ['out_paths', 'out_custom_formats_dict']

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Display.py:633, in PipelineWithDisplayPipelineStageMixin.display(self, display_function, active_session_configuration_context, **kwargs)
    628 if not hasattr(active_session_configuration_context, 'filter_name'):
    629     ## Global session-level context (not filtered, so not corresponding to a specific config name):
    630     ## For a global-style display function, pass ALL of the computation_results and active_configs just to preserve the argument style.
    631     # NOTE: global-style display functions have re-arranged arguments of the form (owning_pipeline_reference, global_computation_results, computation_results, active_configs, **kwargs). This differs from standard ones.
    632     assert getattr(display_function, 'is_global', False), f"display_function must be global if `active_session_configuration_context` does not have a `filter_name` property, but it is not!\n\tdisplay_function:{display_function}\n\tactive_session_configuration_context: {active_session_configuration_context}"
--> 633     curr_display_output = display_function(self, self.global_computation_results, self.computation_results, self.active_configs, active_config_name=None, **kwargs) # CALL GLOBAL DISPLAY FUNCTION
    635 else:
    636     ## Non-global (filtered) context:
    637     # Should be a display functions: The expected filtered context:
    638 
    639     ## Sanity checking:
    640     assert active_session_configuration_name is not None # not true for global contexts

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py:2837, in EpochComputationDisplayFunctions._display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist, save_figure, override_fig_man, ax, custom_export_formats, parent_output_folder, time_bin_size, delete_previous_outputs_folder, desired_height, masked_time_bin_fill_type, enable_ripple_merged_export, enable_laps_merged_export, force_recompute, debug_print, **kwargs)
   2827     # global_dropped_keys, local_dropped_keys = curr_active_pipeline.perform_drop_computed_result(computed_data_keys_to_drop = ['DirectionalDecodersDecoded'], debug_print=True)
   2828 
   2829     ## resolve_and_execute expands prereqs (e.g. DirectionalLaps / DirectionalMergedDecoders) and maps computation_kwargs_list onto matching target names only.
   2830     # owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=['directional_decoders_decode_continuous'],
   2831     #                                       computation_kwargs_list=[{'time_bin_size': time_bin_size, 'should_disable_cache':False}], 
   2832     #                                       enabled_filter_names=None, fail_on_exception=True, debug_print=False)
   2836 print(f'\tcomputing required decoded results at time_bin_size: {time_bin_size} before plotting...')
-> 2837 owning_pipeline_reference.resolve_and_execute_full_required_computation_plan(computation_functions_name_includelist=computation_functions_name_includelist,
   2838                                       computation_kwargs_dict=computation_kwargs_dict,
   2839                                       enabled_filter_names=None, fail_on_exception=True, force_recompute=force_recompute, debug_print=debug_print)
   2842 # owning_pipeline_reference.perform_specific_computation(computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_dict=computation_kwargs_dict,
   2843 #                                 enabled_filter_names=None, fail_on_exception=True, debug_print=debug_print)
   2845 print(f'\t\tdone computing.')

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:58, in stage_wrapper_method.<locals>.wrapper(self, *args, **kwargs)
     56 @functools.wraps(func)
     57 def wrapper(self, *args, **kwargs):
---> 58     return getattr(self.stage, func.__name__)(*args, **kwargs)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1201, in ComputedPipelineStage.resolve_and_execute_full_required_computation_plan(self, active_computation_params, enabled_filter_names, computation_functions_name_includelist, computation_kwargs_list, computation_kwargs_dict, fail_on_exception, debug_print, progress_logger_callback, force_recompute, **kwargs)
   1198     progress_logger_callback(f'\thave {len(functions_to_run)} functions to compute: {functions_to_run}. Performing specific computations: ....')
   1200 aligned_kwargs_list = self.build_computation_kwargs_list_for_function_names(functions_to_run, requested_computation_functions_name_includelist=computation_functions_name_includelist, requested_computation_kwargs_list=computation_kwargs_list, optional_computation_kwargs_dict=computation_kwargs_dict)
-> 1201 self.perform_specific_computation(computation_functions_name_includelist=functions_to_run, computation_kwargs_list=aligned_kwargs_list, enabled_filter_names=enabled_filter_names, fail_on_exception=fail_on_exception, debug_print=debug_print)
   1202 if debug_print:
   1203     progress_logger_callback(f'done.')

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1308, in ComputedPipelineStage.perform_specific_computation(self, active_computation_params, enabled_filter_names, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, debug_print, progress_logger_callback, enable_parallel)
   1306     global_kwargs = dict(owning_pipeline_reference=self, global_computation_results=previous_computation_result, computation_results=self.computation_results, active_configs=self.active_configs, include_includelist=enabled_filter_names, debug_print=debug_print)
   1307     print(f'for global computations: Performing run_specific_computations_single_context(..., computation_functions_name_includelist={computation_functions_name_includelist}, ...)...')
-> 1308     self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
   1309 else:
   1310     # Non-global functions:
   1311     if not enable_parallel:
   1312         ## enable_parallel == False

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:603, in ComputedPipelineStage.run_specific_computations_single_context(self, previous_computation_result, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
    600     progress_logger_callback(f'\trun_specific_computations_single_context(including only {len(active_found_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions): active_computation_functions: {active_found_computation_functions}...')
    602 assert len(active_found_computation_kwargs_list) == len(active_found_computation_functions), f"Length mismatch between computation kwargs list ({len(active_found_computation_kwargs_list)}) and computation functions ({len(active_found_computation_functions)})"        # Perform the computations:
--> 603 return ComputedPipelineStage._execute_computation_functions(active_found_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=active_found_computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1498, in ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
   1496 if progress_logger_callback is not None:
   1497     progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
-> 1498 previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here ## #TODO 2025-02-19 13:51: - [ ] was getting`TypeError: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation() argument after ** must be a mapping, not NoneType` which I fixed by replacing any None in the list with {}
   1499 # Log the computation copmlete time:
   1500 computation_times[computation_times_key_fn(f)] = datetime.now()

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\MultiContextComputationFunctions.py:19, in _wrap_multi_context_computation_function.<locals>._(x, **kwargs)
     16 @wraps(global_comp_fcn) # @wraps ensures that the functions name, docs, etc are accessible in the wrapped version of the function.
     17 def _(x, **kwargs):
     18     assert len(x) > 4, f"looks like it ensures we have more than four (at least 5) positional arguments provided. {x}"
---> 19     x[1] = global_comp_fcn(*x, **kwargs) # update global_computation_results
     20     return x

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py:2181, in EpochComputationFunctions.perform_generalized_specific_epochs_decoding(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist, debug_print, epochs_decoding_time_bin_size, drop_previous_result_and_compute_fresh, force_recompute)
   2177         print(f'removed previous "EpochComputations.a_generic_decoder_dict_decoded_epochs_dict_result" result and computing fresh since `drop_previous_result_and_compute_fresh == True`')
   2179 if (not hasattr(valid_EpochComputations_result, 'a_generic_decoder_dict_decoded_epochs_dict_result')) or (getattr(valid_EpochComputations_result, 'a_generic_decoder_dict_decoded_epochs_dict_result', None) is None):
   2180     # initialize
-> 2181     a_new_fully_generic_result: GenericDecoderDictDecodedEpochsDictResult = GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(curr_active_pipeline=owning_pipeline_reference, force_recompute=force_recompute, time_bin_size=epochs_decoding_time_bin_size, debug_print=debug_print)
   2182     valid_EpochComputations_result.a_generic_decoder_dict_decoded_epochs_dict_result = a_new_fully_generic_result
   2183     global_computation_results.computed_data['EpochComputations'].a_generic_decoder_dict_decoded_epochs_dict_result = a_new_fully_generic_result

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\context_dependent.py:1141, in GenericDecoderDictDecodedEpochsDictResult.batch_user_compute_fn(cls, curr_active_pipeline, force_recompute, time_bin_size, debug_print)
   1137     _perform_comp_kwargs = dict(fail_on_exception=True)
   1139 ## perform the computation either way:
   1140 # curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['non_PBE_epochs_results'], enabled_filter_names=None, fail_on_exception=True, debug_print=False)
-> 1141 curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=['merged_directional_placefields', 'directional_decoders_decode_continuous', 'directional_decoders_evaluate_epochs', 'directional_decoders_epoch_heuristic_scoring', 'non_PBE_epochs_results'],
   1142                                                 computation_kwargs_list=[{'ripple_decoding_time_bin_size': time_bin_size, 'laps_decoding_time_bin_size': time_bin_size}, {'time_bin_size': time_bin_size}, {'should_skip_radon_transform': True},
   1143                                                                             {'same_thresh_fraction_of_track': 0.05, 'max_ignore_bins': 2, 'use_bin_units_instead_of_realworld': False, 'max_jump_distance_cm': 60.0},
   1144                                                                              dict(epochs_decoding_time_bin_size=time_bin_size, frame_divide_bin_size=10.0, compute_1D=True, compute_2D=True, drop_previous_result_and_compute_fresh=force_recompute, skip_training_test_split=True, debug_print_memory_breakdown=False),
   1145                                                                         ], ## END KWARGS LIST
   1146                                                                         enabled_filter_names=None, debug_print=False, **_perform_comp_kwargs)
   1147 curr_active_pipeline.batch_extended_computations(include_includelist=['non_PBE_epochs_results'], include_global_functions=True, included_computation_filter_names=None, fail_on_exception=True, debug_print=False) ## just checking
   1150 session_name: str = curr_active_pipeline.session_name

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1308, in ComputedPipelineStage.perform_specific_computation(self, active_computation_params, enabled_filter_names, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, debug_print, progress_logger_callback, enable_parallel)
   1306     global_kwargs = dict(owning_pipeline_reference=self, global_computation_results=previous_computation_result, computation_results=self.computation_results, active_configs=self.active_configs, include_includelist=enabled_filter_names, debug_print=debug_print)
   1307     print(f'for global computations: Performing run_specific_computations_single_context(..., computation_functions_name_includelist={computation_functions_name_includelist}, ...)...')
-> 1308     self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print, progress_logger_callback=progress_logger_callback) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
   1309 else:
   1310     # Non-global functions:
   1311     if not enable_parallel:
   1312         ## enable_parallel == False

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:603, in ComputedPipelineStage.run_specific_computations_single_context(self, previous_computation_result, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
    600     progress_logger_callback(f'\trun_specific_computations_single_context(including only {len(active_found_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions): active_computation_functions: {active_found_computation_functions}...')
    602 assert len(active_found_computation_kwargs_list) == len(active_found_computation_functions), f"Length mismatch between computation kwargs list ({len(active_found_computation_kwargs_list)}) and computation functions ({len(active_found_computation_functions)})"        # Perform the computations:
--> 603 return ComputedPipelineStage._execute_computation_functions(active_found_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=active_found_computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py:1498, in ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
   1496 if progress_logger_callback is not None:
   1497     progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
-> 1498 previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here ## #TODO 2025-02-19 13:51: - [ ] was getting`TypeError: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation() argument after ** must be a mapping, not NoneType` which I fixed by replacing any None in the list with {}
   1499 # Log the computation copmlete time:
   1500 computation_times[computation_times_key_fn(f)] = datetime.now()

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\MultiContextComputationFunctions.py:19, in _wrap_multi_context_computation_function.<locals>._(x, **kwargs)
     16 @wraps(global_comp_fcn) # @wraps ensures that the functions name, docs, etc are accessible in the wrapped version of the function.
     17 def _(x, **kwargs):
     18     assert len(x) > 4, f"looks like it ensures we have more than four (at least 5) positional arguments provided. {x}"
---> 19     x[1] = global_comp_fcn(*x, **kwargs) # update global_computation_results
     20     return x

File H:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\EpochComputationFunctions.py:1994, in EpochComputationFunctions.perform_compute_non_PBE_epochs(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist, debug_print, training_data_portion, epochs_decoding_time_bin_size, frame_divide_bin_size, compute_1D, compute_2D, drop_previous_result_and_compute_fresh, skip_training_test_split, debug_print_memory_breakdown, IGNORE_MEMORY_ERROR_FOR_DEBUGGING)
   1992         print(f'WARNING: IGNORE_MEMORY_ERROR_FOR_DEBUGGING == True so the MemoryError exception will be suppressed, and computation will unwisely continue! memory_error_msg: {memory_error_msg}')
   1993     else:
-> 1994         raise MemoryError(memory_error_msg)
   1995     # return global_computation_results
   1996     
   1997 # ==================================================================================================================== #
   1998 # Proceed with computation                                                                                             #
   1999 # ==================================================================================================================== #
   2000 if include_includelist is not None:

MemoryError: Estimated Insufficient Memory: Operation would require 14.00 GB (have 12.66 GB available.
```
