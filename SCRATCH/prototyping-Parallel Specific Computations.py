from collections import OrderedDict
import sys
from copy import deepcopy
from datetime import datetime, timedelta
import typing
from typing import Callable, Optional, Dict, List, Tuple, Union
from warnings import warn
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum # for EvaluationActions
from datetime import datetime
from attrs import define, field, Factory
from pyphoplacecellanalysis.General.Model.ComputationResults import ComputationResult
from pyphoplacecellanalysis.General.Pipeline.Stages.Computation import ComputedPipelineStage, FunctionsSearchMode


class ParallelComputationHelper:
    """ Based off of "specific" computation functions in `_execute_computation_functions`: vscode://file/c:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py:771

       unimplemented:
    
    find_registered_computation_functions
    registered_computation_function_names

    
    Should run a SINGLE computation, and should be executable in parallel.

    Common:
        active_computation_fn,

    If Global:

    If Non-Global:
        computation_context




    """
    
    def run_specific_computations_single_context(self, previous_computation_result, computation_functions_name_includelist, computation_kwargs_list=None, fail_on_exception:bool=False, progress_logger_callback=None, are_global:bool=False, debug_print=False):
        """ re-runs just a specific computation provided by computation_functions_name_includelist """
        active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, search_mode=FunctionsSearchMode.initFromIsGlobal(are_global))
        if progress_logger_callback is not None:
            progress_logger_callback(f'run_specific_computations_single_context(including only {len(active_computation_functions)} out of {len(self.registered_computation_function_names)} registered computation functions): active_computation_functions: {active_computation_functions}...')
        # Perform the computations:
        return ComputedPipelineStage._execute_computation_functions(active_computation_functions, previous_computation_result=previous_computation_result, computation_kwargs_list=computation_kwargs_list, fail_on_exception=fail_on_exception, progress_logger_callback=progress_logger_callback, are_global=are_global, debug_print=debug_print)


    def perform_specific_computation(self, active_computation_params=None, enabled_filter_names=None, computation_functions_name_includelist=None, computation_kwargs_list=None, fail_on_exception:bool=False, debug_print=False):
        """ perform a specific computation (specified in computation_functions_name_includelist) in a minimally destructive manner using the previously recomputed results:
        Ideally would already have access to the:
        - Previous computation result
        - Previous computation config (the input parameters)


        computation_kwargs_list: Optional<list>: a list of kwargs corresponding to each function name in computation_functions_name_includelist

        Internally calls: `run_specific_computations_single_context`.

        Updates:
            curr_active_pipeline.computation_results
        """
        if enabled_filter_names is None:
            enabled_filter_names = list(self.filtered_sessions.keys()) # all filters if specific enabled names aren't specified

        has_custom_kwargs_list: bool = False # indicates whether user provided a kwargs list
        if computation_kwargs_list is None:
            computation_kwargs_list = [{} for _ in computation_functions_name_includelist]
        else:
            has_custom_kwargs_list = np.any([len(x)>0 for x in computation_kwargs_list])
            # has_custom_kwargs_list = True            

        assert isinstance(computation_kwargs_list, List), f"computation_kwargs_list: Optional<list>: is supposed to be a list of kwargs corresponding to each function name in computation_functions_name_includelist but instead is of type:\n\ttype(computation_kwargs_list): {type(computation_kwargs_list)}"
        assert len(computation_kwargs_list) == len(computation_functions_name_includelist)


        active_computation_functions = self.find_registered_computation_functions(computation_functions_name_includelist, search_mode=FunctionsSearchMode.ANY) # find_registered_computation_functions is a pipeline.stage property
        contains_any_global_functions = np.any([v.is_global for v in active_computation_functions])
        if contains_any_global_functions:
            assert np.all([v.is_global for v in active_computation_functions]), 'ERROR: cannot mix global and non-global functions in a single call to perform_specific_computation'

            if (self.global_computation_results is None) or (not isinstance(self.global_computation_results, ComputationResult)):
                print(f'global_computation_results is None. Building initial global_computation_results...')
                self.global_computation_results = None # clear existing results
                self.global_computation_results = ComputedPipelineStage._build_initial_computationResult(self.sess, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
                

        if contains_any_global_functions:
            # global computation functions:
            if (self.global_computation_results is None) or (not isinstance(self.global_computation_results, ComputationResult)):
                print(f'global_computation_results is None or not a `ComputationResult` object. Building initial global_computation_results...') #TODO 2024-01-10 15:12: - [ ] Check that `self.global_computation_results.keys()` are empty
                self.global_computation_results = None # clear existing results
                self.global_computation_results = ComputedPipelineStage._build_initial_computationResult(self.sess, active_computation_params) # returns a computation result. This stores the computation config used to compute it.
            ## TODO: what is this about?
            previous_computation_result = self.global_computation_results

            ## TODO: ERROR: `owning_pipeline_reference=self` is not CORRECT as self is of type `ComputedPipelineStage` (or `DisplayPipelineStage`) and not `NeuropyPipeline`
                # this has been fine for all the global functions so far because the majority of the properties are defined on the stage anyway, but any pipeline properties will be missing! 
            global_kwargs = dict(owning_pipeline_reference=self, global_computation_results=previous_computation_result, computation_results=self.computation_results, active_configs=self.active_configs, include_includelist=enabled_filter_names, debug_print=debug_print)
            self.global_computation_results = self.run_specific_computations_single_context(global_kwargs, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=True, fail_on_exception=fail_on_exception, debug_print=debug_print) # was there a reason I didn't pass `computation_kwargs_list` to the global version?
        else:
            # Non-global functions:
            for a_select_config_name, a_filtered_session in self.filtered_sessions.items():                
                if a_select_config_name in enabled_filter_names:
                    print(f'Performing run_specific_computations_single_context on filtered_session with filter named "{a_select_config_name}"...')
                    if active_computation_params is None:
                        curr_active_computation_params = self.active_configs[a_select_config_name].computation_config # get the previously set computation configs
                    else:
                        # set/update the computation configs:
                        curr_active_computation_params = active_computation_params 
                        self.active_configs[a_select_config_name].computation_config = curr_active_computation_params #TODO: if more than one computation config is passed in, the active_config should be duplicated for each computation config.

                    ## Here is an issue, we need to get the appropriate computation result depending on whether it's global or not 
                    previous_computation_result = self.computation_results[a_select_config_name]
                    self.computation_results[a_select_config_name] = self.run_specific_computations_single_context(previous_computation_result, computation_functions_name_includelist=computation_functions_name_includelist, computation_kwargs_list=computation_kwargs_list, are_global=False, fail_on_exception=fail_on_exception, debug_print=debug_print)
        
        ## IMPLEMENTATION FAULT: the global computations/results should not be ran within the filter/config loop. It applies to all config names and should be ran last. Also don't allow mixing local/global functions.
