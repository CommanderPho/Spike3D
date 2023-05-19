""" run_BatchAnalysis.py
2023-05-19 - Updated version of batch analysis from simple Python scripts for use with Data Version Control (DVC) pipelines and dvclive. 

"""
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
from pyphocorehelpers.function_helpers import function_attributes

# pyPhoPlaceCellAnalysis:
# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
from neuropy.core.session.Formats.Specific.HiroDataSessionFormat import HiroDataSessionFormatRegisteredClass
from neuropy.core.epoch import Epoch

## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths

# from PendingNotebookCode import _perform_batch_plot, _build_batch_plot_kwargs
from pyphoplacecellanalysis.General.Batch.NonInteractiveWrapper import batch_load_session, batch_extended_computations, SessionBatchProgress, batch_programmatic_figures, batch_extended_programmatic_figures
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun
from pyphoplacecellanalysis.General.Batch.runBatch import run_diba_batch
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests
from dvclive import Live

with Live(save_dvc_exp=True) as live:

	live.log_param("epochs", 5)
        
    def post_compute_validate(curr_active_pipeline):
        """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
        LongShortPipelineTests(curr_active_pipeline=curr_active_pipeline).validate()


    def _on_complete_success_execution_session(curr_session_context, curr_session_basedir, curr_active_pipeline):
        """ called when the execute_session completes like:
            `post_run_callback_fn_output = post_run_callback_fn(curr_session_context, curr_session_basedir, curr_active_pipeline)`
            
            Meant to be assigned like:
            , post_run_callback_fn=_on_complete_success_execution_session
            
            Captures nothing.
        """
        print(f'_on_complete_success_execution_session(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...)')
        # print(f'curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}')
        long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
        # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # long_results, short_results, global_results = [curr_active_pipeline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]

        # Get existing laps from session:
        long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
        # short_laps.n_epochs: 40, long_laps.n_epochs: 40
        # short_replays.n_epochs: 6, long_replays.n_epochs: 8
        print(f'short_laps.n_epochs: {short_laps.n_epochs}, long_laps.n_epochs: {long_laps.n_epochs}')
        print(f'short_replays.n_epochs: {short_replays.n_epochs}, long_replays.n_epochs: {long_replays.n_epochs}')

        ## Post Compute Validate 2023-05-16:
        post_compute_validate(curr_active_pipeline)

        # # 2023-01-* - Call extended computations to build `_display_short_long_firing_rate_index_comparison` figures:
        # extended_computations_include_whitelist=['long_short_fr_indicies_analyses', 'jonathan_firing_rate_analysis', 'long_short_decoding_analyses'] # do only specifiedl
        # # extended_computations_include_whitelist=['long_short_fr_indicies_analyses', 'jonathan_firing_rate_analysis'] # do only specifiedl
        # newly_computed_values = batch_extended_computations(curr_active_pipeline, include_whitelist=extended_computations_include_whitelist, include_global_functions=True, fail_on_exception=True, progress_print=True, force_recompute=True, debug_print=False)
        # curr_active_pipeline.save_global_computation_results()

        # ### Programmatic Figure Outputs:
        # # Other Programmatic Figures
        # batch_extended_programmatic_figures(curr_active_pipeline=curr_active_pipeline)
        # batch_programmatic_figures(curr_active_pipeline=curr_active_pipeline)

        return {long_epoch_name:(long_laps, long_replays), short_epoch_name:(short_laps, short_replays)}



    def main(active_global_batch_result_filename='global_batch_result.pkl', debug_print=True):
        """ Main Run Function
        from pyphoplacecellanalysis.General.Batch.runBatch import main, BatchRun, run_diba_batch, run_specific_batch

        """
        global_data_root_parent_path = find_first_extant_path([Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data')])
        assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
        ## Build Pickle Path:
        finalized_loaded_global_batch_result_pickle_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve() # Use Default

        ## Override:
        # finalized_loaded_global_batch_result_pickle_path = Path('data/global_batch_run-2023-05-17.pkl').resolve()

        def _try_load_global_batch_result():
            """ load the batch result initially. 
            Captures: finalized_loaded_global_batch_result_pickle_path, debug_print
            """
            if debug_print:
                print(f'finalized_loaded_global_batch_result_pickle_path: {finalized_loaded_global_batch_result_pickle_path}')
            # try to load an existing batch result:
            try:
                global_batch_run = loadData(finalized_loaded_global_batch_result_pickle_path, debug_print=debug_print)
                
            except NotImplementedError:
                # Fixes issue with pickled POSIX_PATH on windows for path.
                posix_backup = pathlib.PosixPath # backup the PosixPath definition
                try:
                    pathlib.PosixPath = pathlib.PurePosixPath
                    global_batch_run = loadData(finalized_loaded_global_batch_result_pickle_path, debug_print=debug_print) # Fails this time if it still throws an error
                finally:
                    pathlib.PosixPath = posix_backup # restore the backup posix path definition
                    
            except (FileNotFoundError, TypeError):
                # loading failed
                print(f'Failure loading {finalized_loaded_global_batch_result_pickle_path}.')
                global_batch_run = None
                
            return global_batch_run

        ##

        global_batch_run = _try_load_global_batch_result()
        if global_batch_run is not None:
            # One was loaded from file, meaning it has the potential to have the wrong paths. Check.
            global_batch_run.change_global_root_path(global_data_root_parent_path) # Convert the paths to work on the new system:
        else:
            ## Completely fresh, run the initial (pre-loading) results.
            # Build `global_batch_run` pre-loading results (before execution)
            global_batch_run = run_diba_batch(global_data_root_parent_path, execute_all=False, extant_batch_run=global_batch_run, debug_print=False)
            # print(f'global_batch_result: {global_batch_run}')
            # Save `global_batch_run` to file:
            saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary

        ## I got it doing the bare-minimum loading and computations, so it should be ready to update the laps and constrain the placefields to those. Then we should be able to set up the replays at the same time.
        # finally, we then finish by computing.
        assert global_batch_run is not None


        # Run Batch Executions/Computations
        

        ## Execute the non-global functions with the custom arguments.
        active_computation_functions_name_whitelist=['_perform_baseline_placefield_computation',
                                                # '_perform_time_dependent_placefield_computation',
                                                '_perform_extended_statistics_computation',
                                                '_perform_position_decoding_computation', 
                                                '_perform_firing_rate_trends_computation',
                                                '_perform_pf_find_ratemap_peaks_computation',
                                                # '_perform_time_dependent_pf_sequential_surprise_computation'
                                                # '_perform_two_step_position_decoding_computation',
                                                # '_perform_recursive_latent_placefield_decoding'
                                            ]
        global_batch_run.execute_all(force_reload=False, skip_extended_batch_computations=True, post_run_callback_fn=_on_complete_success_execution_session,
                            **{'computation_functions_name_whitelist': active_computation_functions_name_whitelist,
                            'active_session_computation_configs': None}) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)
        # 4m 39.8s


        # Save `global_batch_run` to file:
        saveData(finalized_loaded_global_batch_result_pickle_path, global_batch_run) # Update the global batch run dictionary
        
        # for epoch in range(NUM_EPOCHS):
        #     train_model(...)
        #     metrics = evaluate_model(...)

        #     for metric_name, value in metrics.items():
        #         live.log_metric(metric_name, value)

        #     live.next_step()

        params = {
            "num_classes": 10,
            "metrics": ["accuracy", "mae"],
            "optimizer": "adam"
        }
        live.log_params(params)
            
        return global_batch_run, finalized_loaded_global_batch_result_pickle_path

        


        # ### Get Outputs
        # batch_progress_df = global_batch_run.to_dataframe(expand_context=True) # all
        # good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
        # good_only_batch_progress_df

if __name__ == "__main__":
    """ run main function to perform batch processing. """
    global_batch_run, finalized_loaded_global_batch_result_pickle_path = main(active_global_batch_result_filename='global_batch_result_new.pkl', debug_print=True)