{
	"name": "ZeroDivisionError",
	"message": "float division by zero",
	"stack": "---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Model\\SpecificComputationValidation.py:216, in SpecificComputationValidator._perform_try_computation_if_needed(cls, comp_specifier, curr_active_pipeline, computation_filter_name, on_already_computed_fn, fail_on_exception, progress_print, debug_print, force_recompute)
    215 if (is_known_missing_provided_keys):
--> 216     raise ValueError(f\"missing required value, so we don't need to call .validate_computation_test(...) to know it isn't valid!\")
    217 else:

ValueError: missing required value, so we don't need to call .validate_computation_test(...) to know it isn't valid!

During handling of the above exception, another exception occurred:

ZeroDivisionError                         Traceback (most recent call last)
c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\ReviewOfWork_2024-01-08_FRESH.ipynb Cell 4 line 3
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-08_FRESH.ipynb#W3sZmlsZQ%3D%3D?line=29'>30</a> force_recompute_global = force_reload
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-08_FRESH.ipynb#W3sZmlsZQ%3D%3D?line=30'>31</a> # force_recompute_global = True
---> <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-08_FRESH.ipynb#W3sZmlsZQ%3D%3D?line=31'>32</a> newly_computed_values = batch_extended_computations(curr_active_pipeline, include_includelist=extended_computations_include_includelist, include_global_functions=True, fail_on_exception=True, progress_print=True,
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-08_FRESH.ipynb#W3sZmlsZQ%3D%3D?line=32'>33</a>                                                     force_recompute=force_recompute_global, force_recompute_override_computations_includelist=force_recompute_override_computations_includelist, debug_print=False)
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-08_FRESH.ipynb#W3sZmlsZQ%3D%3D?line=33'>34</a> if (len(newly_computed_values) > 0):
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2024-01-08_FRESH.ipynb#W3sZmlsZQ%3D%3D?line=34'>35</a>     print(f'newly_computed_values: {newly_computed_values}.')

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Batch\\NonInteractiveProcessing.py:355, in batch_extended_computations(curr_active_pipeline, include_includelist, included_computation_filter_names, include_global_functions, fail_on_exception, progress_print, debug_print, force_recompute, force_recompute_override_computations_includelist, dry_run)
    353 _curr_force_recompute = force_recompute or ((_comp_specifier.short_name in force_recompute_override_computations_includelist) or (_comp_specifier.computation_fn_name in force_recompute_override_computations_includelist)) # force_recompute for this specific result if either of its name is included in `force_recompute_override_computations_includelist`
    354 if not dry_run:
--> 355     newly_computed_values += _comp_specifier.try_computation_if_needed(curr_active_pipeline, computation_filter_name=global_epoch_name, on_already_computed_fn=_subfn_on_already_computed, fail_on_exception=fail_on_exception, progress_print=progress_print, debug_print=debug_print, force_recompute=_curr_force_recompute)
    356 else:
    357     print(f'dry-run: {_comp_specifier.short_name}, force_recompute={force_recompute}, curr_force_recompute={_curr_force_recompute}')

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Model\\SpecificComputationValidation.py:139, in SpecificComputationValidator.try_computation_if_needed(self, curr_active_pipeline, **kwargs)
    138 def try_computation_if_needed(self, curr_active_pipeline, **kwargs):
--> 139     return self._perform_try_computation_if_needed(self, curr_active_pipeline, **kwargs)

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Model\\SpecificComputationValidation.py:232, in SpecificComputationValidator._perform_try_computation_if_needed(cls, comp_specifier, curr_active_pipeline, computation_filter_name, on_already_computed_fn, fail_on_exception, progress_print, debug_print, force_recompute)
    230 # When this fails due to unwrapping from the load, add `, computation_kwargs_list=[{'perform_cache_load': False}]` as an argument to the `perform_specific_computation` call below
    231 try:
--> 232     curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=[comp_specifier.computation_fn_name], computation_kwargs_list=[comp_specifier.computation_fn_kwargs], fail_on_exception=True, debug_print=False) # fail_on_exception MUST be True or error handling is all messed up 
    233     if progress_print or debug_print:
    234         print(f'\\t done.')

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Computation.py:1074, in PipelineWithComputedPipelineStageMixin.perform_specific_computation(self, active_computation_params, enabled_filter_names, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, debug_print)
   1067 \"\"\" perform a specific computation (specified in computation_functions_name_includelist) in a minimally destructive manner using the previously recomputed results:
   1068 Passthrough wrapper to self.stage.perform_specific_computation(...) with the same arguments.
   1069 
   1070 Updates:
   1071     curr_active_pipeline.computation_results
   1072 \"\"\"
   1073 # self.stage is of type ComputedPipelineStage
-> 1074 return self.stage.perform_specific_computation(active_computation_params=active_computation_params, enabled_filter_names=enabled_filter_names, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, fail_on_exception=fail_on_exception, debug_print=debug_print)

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Computation.py:666, in ComputedPipelineStage.perform_specific_computation(self, active_computation_params, enabled_filter_names, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, debug_print)
    662     global_kwargs = dict(owning_pipeline_reference=self, global_computation_results=previous_computation_result, computation_results=self.computation_results, active_configs=self.active_configs, include_includelist=enabled_filter_names, debug_print=debug_print)
    664     assert (not has_custom_kwargs_list), f\"#TODO 2023-11-22 23:41: - [ ] perform_specific_computation(...) computation_kwargs_list seems to have no effect in global functions, maybe fix? For now, just throw an error so you don't think your custom kwargs are working when they aren't.\"
--> 666     self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
    667 else:
    668     # Non-global functions:
    669     for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Computation.py:330, in ComputedPipelineStage.run_specific_computations_single_context(self, previous_computation_result, computation_functions_name_includelist, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
    328     progress_logger_callback(f'run_specific_computations_single_context(including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions): active_computation_functions: {active_computation_functions}...')
    329 # Perform the computations:
--> 330 return ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Computation.py:771, in ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result, computation_kwargs_list, fail_on_exception, progress_logger_callback, are_global, debug_print)
    769 if progress_logger_callback is not None:
    770     progress_logger_callback(f'Executing [{i}/{total_num_funcs}]: {f}')
--> 771 previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i])
    772 # Log the computation copmlete time:
    773 computation_times[f] = datetime.now()

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\MultiContextComputationFunctions.py:18, in _wrap_multi_context_computation_function.<locals>._(x)
     15 @wraps(global_comp_fcn) # @wraps ensures that the functions name, docs, etc are accessible in the wrapped version of the function.
     16 def _(x):
     17     assert len(x) > 4, f\"{x}\"
---> 18     x[1] = global_comp_fcn(*x) # update global_computation_results
     19     return x

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\RankOrderComputations.py:2683, in RankOrderGlobalComputationFunctions.perform_rank_order_shuffle_analysis(owning_pipeline_reference, global_computation_results, computation_results, active_configs, include_includelist, debug_print, num_shuffles, minimum_inclusion_fr_Hz, included_qclu_values, skip_laps)
   2681     print(f'\\tdone. building global result.')
   2682     global_computation_results.computed_data['RankOrder'].adding_active_aclus_info()
-> 2683     global_computation_results.computed_data['RankOrder'].ripple_most_likely_result_tuple, global_computation_results.computed_data['RankOrder'].laps_most_likely_result_tuple = RankOrderAnalyses.most_likely_directional_rank_order_shuffling(owning_pipeline_reference)
   2685 except (AssertionError, BaseException) as e:
   2686     print(f'Issue with `RankOrderAnalyses.most_likely_directional_rank_order_shuffling(...)` e: {e}')

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\RankOrderComputations.py:2081, in RankOrderAnalyses.most_likely_directional_rank_order_shuffling(cls, curr_active_pipeline)
   2077     laps_result_tuple = None
   2080 # Compute the quantiles:
-> 2081 cls.percentiles_computations(rank_order_results=rank_order_results)
   2083 return ripple_result_tuple, laps_result_tuple

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\RankOrderComputations.py:1818, in RankOrderAnalyses.percentiles_computations(cls, rank_order_results)
   1811 # # `LongShortStatsItem` form (2024-01-02):        
   1812 # new_LR_results_quantile_values = np.array([(compute_percentile(a_result_item.long_stats_z_scorer.real_value, a_result_item.long_stats_z_scorer.original_values), compute_percentile(a_result_item.short_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.original_values)) for epoch_id, a_result_item in rank_order_results.LR_laps.ranked_aclus_stats_dict.items()])
   1813 # new_RL_results_quantile_values = np.array([(compute_percentile(a_result_item.long_stats_z_scorer.real_value, a_result_item.long_stats_z_scorer.original_values), compute_percentile(a_result_item.short_stats_z_scorer.real_value, a_result_item.short_stats_z_scorer.original_values)) for epoch_id, a_result_item in rank_order_results.RL_laps.ranked_aclus_stats_dict.items()])
   1814 
   1815 ## 2023-12-23 Method:        
   1816 # recover from the valid stacked arrays: `valid_stacked_arrays`
   1817 output_active_epoch_computed_values, combined_variable_names, valid_stacked_arrays, real_stacked_arrays, n_valid_shuffles = rank_order_results.laps_new_output_tuple      
-> 1818 quantile_results_dict_laps = compute_percentiles_from_shuffle_results(combined_variable_names, valid_stacked_arrays, real_stacked_arrays)
   1820 # new_LR_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(shuffled_results_output_dict['long_LR_pearson_Z'][0], shuffled_results_output_dict['short_LR_pearson_Z'][0])])
   1821 # new_RL_results_quantile_values = np.array([(compute_percentile(long_stats_z_scorer.real_value, long_stats_z_scorer.original_values), compute_percentile(short_stats_z_scorer.real_value, short_stats_z_scorer.original_values)) for long_stats_z_scorer, short_stats_z_scorer in zip(shuffled_results_output_dict['short_LR_pearson_Z'][0], shuffled_results_output_dict['short_RL_pearson_Z'][0])])
   1822 # quantile_results_dict = dict(zip(['LR_Long_percentile', 'LR_Short_percentile', 'RL_Long_percentile', 'RL_Short_percentile'], np.hstack((new_LR_results_quantile_values, new_RL_results_quantile_values)).T))
   1823 # quantile_results_df = pd.DataFrame(np.hstack((new_LR_results_real_values, new_RL_results_real_values)), columns=['LR_Long_percentile', 'LR_Short_percentile', 'RL_Long_percentile', 'RL_Short_percentile'])
   1824 
   1825 ## Add the new columns into the `laps_combined_epoch_stats_df`
   1826 for a_col_name, col_vals in quantile_results_dict_laps.items():

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\RankOrderComputations.py:1749, in RankOrderAnalyses.percentiles_computations.<locals>.compute_percentiles_from_shuffle_results(combined_variable_names, valid_stacked_arrays, real_stacked_arrays)
   1747     assert n_variables == np.shape(valid_stacked_arrays)[-1]\t\t
   1748     a_result_column_name: str = quantile_result_column_names[variable_IDX] # column name with the suffix '_percentile' added to it
-> 1749     results_quantile_value[a_result_column_name] = np.array([compute_percentile(real_stacked_arrays[epoch_IDX, variable_IDX], np.squeeze(valid_stacked_arrays[:, epoch_IDX, variable_IDX])) for epoch_IDX in np.arange(n_epochs)]) # real_stacked_arrays based version
   1750     # results_quantile_value[a_column_name] = np.array([compute_percentile(real_values[epoch_IDX], np.squeeze(valid_stacked_arrays[:, epoch_IDX, variable_IDX])) for epoch_IDX in np.arange(n_epochs)]) # working df-based version
   1751 
   1752 # Add old columns for compatibility:
   1753 for old_col_name, new_col_name in zip(['LR_Long_percentile', 'RL_Long_percentile', 'LR_Short_percentile', 'RL_Short_percentile'], ['LR_Long_pearson_percentile', 'RL_Long_pearson_percentile', 'LR_Short_pearson_percentile', 'RL_Short_pearson_percentile']):

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\RankOrderComputations.py:1749, in <listcomp>(.0)
   1747     assert n_variables == np.shape(valid_stacked_arrays)[-1]\t\t
   1748     a_result_column_name: str = quantile_result_column_names[variable_IDX] # column name with the suffix '_percentile' added to it
-> 1749     results_quantile_value[a_result_column_name] = np.array([compute_percentile(real_stacked_arrays[epoch_IDX, variable_IDX], np.squeeze(valid_stacked_arrays[:, epoch_IDX, variable_IDX])) for epoch_IDX in np.arange(n_epochs)]) # real_stacked_arrays based version
   1750     # results_quantile_value[a_column_name] = np.array([compute_percentile(real_values[epoch_IDX], np.squeeze(valid_stacked_arrays[:, epoch_IDX, variable_IDX])) for epoch_IDX in np.arange(n_epochs)]) # working df-based version
   1751 
   1752 # Add old columns for compatibility:
   1753 for old_col_name, new_col_name in zip(['LR_Long_percentile', 'RL_Long_percentile', 'LR_Short_percentile', 'RL_Short_percentile'], ['LR_Long_pearson_percentile', 'RL_Long_pearson_percentile', 'LR_Short_pearson_percentile', 'RL_Short_pearson_percentile']):

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\ComputationFunctions\\MultiContextComputationFunctions\\RankOrderComputations.py:1710, in RankOrderAnalyses.percentiles_computations.<locals>.compute_percentile(real_value, original_shuffle_values)
   1709 def compute_percentile(real_value, original_shuffle_values):
-> 1710     return (1.0 - float(np.sum((np.abs(real_value) < original_shuffle_values)))/float(len(original_shuffle_values)))

ZeroDivisionError: float division by zero"
}