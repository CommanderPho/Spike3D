Performing _execute_computation_functions(...) with 5 registered_computation_functions...
Performing _execute_computation_functions(...) with 5 registered_computation_functions...
Executing [0/5]: <function PlacefieldComputations._perform_baseline_placefield_computation at 0x000001995730EDC0>
Unexpected exception formatting exception. Falling back to standard exception
Traceback (most recent call last):
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\interactiveshell.py", line 3550, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "C:\Users\pho\AppData\Local\Temp\ipykernel_31568\4049252502.py", line 19, in <module>
    _across_session_results_extended_dict = _across_session_results_extended_dict | kdiba_session_post_fixup_completion_function(a_dummy, None,
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py", line 2522, in kdiba_session_post_fixup_completion_function
    across_session_results_extended_dict = PostHocPipelineFixup.run_as_batch_user_completion_function(self=self, global_data_root_parent_path=global_data_root_parent_path, curr_session_context=curr_session_context, curr_session_basedir=curr_session_basedir, curr_active_pipeline=curr_active_pipeline, across_session_results_extended_dict=across_session_results_extended_dict, force_recompute=force_recompute)
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py", line 2465, in run_as_batch_user_completion_function
    (did_any_grid_bin_change, change_dict), correct_grid_bin_bounds = PostHocPipelineFixup.FINAL_FIX_GRID_BIN_BOUNDS(curr_active_pipeline=curr_active_pipeline, force_recompute=force_recompute, is_dry_run=False)
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py", line 2418, in FINAL_FIX_GRID_BIN_BOUNDS
    curr_active_pipeline.perform_specific_computation(computation_functions_name_includelist=computation_functions_name_includelist, fail_on_exception=True, debug_print=True)
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 2040, in perform_specific_computation
    if (not is_dry_run):
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 907, in perform_specific_computation
    progress_logger_callback = print
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 425, in run_specific_computations_single_context
  File "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Computation.py", line 1171, in _execute_computation_functions
    previous_computation_result = f(previous_computation_result, **computation_kwargs_list[i]) # call the function `f` directly here
TypeError: pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.PlacefieldComputations.PlacefieldComputations._perform_baseline_placefield_computation() argument after ** must be a mapping, not NoneType

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\interactiveshell.py", line 2144, in showtraceback
    stb = self.InteractiveTB.structured_traceback(
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\ultratb.py", line 1435, in structured_traceback
    return FormattedTB.structured_traceback(
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\ultratb.py", line 1326, in structured_traceback
    return VerboseTB.structured_traceback(
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\ultratb.py", line 1173, in structured_traceback
    formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\ultratb.py", line 1088, in format_exception_as_a_whole
    frames.append(self.format_record(record))
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\ultratb.py", line 970, in format_record
    frame_info.lines, Colors, self.has_colors, lvals
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\IPython\core\ultratb.py", line 792, in lines
    return self._sd.lines
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\stack_data\utils.py", line 145, in cached_property_wrapper
    value = obj.__dict__[self.func.__name__] = self.func(obj)
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\stack_data\core.py", line 734, in lines
    pieces = self.included_pieces
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\stack_data\utils.py", line 145, in cached_property_wrapper
    value = obj.__dict__[self.func.__name__] = self.func(obj)
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\stack_data\core.py", line 681, in included_pieces
    pos = scope_pieces.index(self.executing_piece)
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\stack_data\utils.py", line 145, in cached_property_wrapper
    value = obj.__dict__[self.func.__name__] = self.func(obj)
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\stack_data\core.py", line 660, in executing_piece
    return only(
  File "c:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\lib\site-packages\executing\executing.py", line 116, in only
    raise NotOneValueFound('Expected one value, found 0')
executing.executing.NotOneValueFound: Expected one value, found 0