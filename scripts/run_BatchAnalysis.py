""" run_BatchAnalysis.py
2023-05-19 - Updated version of batch analysis from simple Python scripts for use with Data Version Control (DVC) pipelines and dvclive. 

"""
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
import neptune # for logging progress and results
from neptune.types import File

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
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_load_session, batch_extended_computations, SessionBatchProgress, batch_programmatic_figures, batch_extended_programmatic_figures
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun
from pyphoplacecellanalysis.General.Batch.runBatch import run_diba_batch
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests
# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots, _update_pipeline_missing_preprocessing_parameters

## Post Compute Validate 2023-05-16:

def post_compute_validate(curr_active_pipeline):
    """ 2023-05-16 - Ensures that the laps are used for the placefield computation epochs, the number of bins are the same between the long and short tracks. """
    LongShortPipelineTests(curr_active_pipeline=curr_active_pipeline).validate()
    # 2023-05-24 - Adds the previously missing `sess.config.preprocessing_parameters` to each session (filtered and base) in the pipeline.
    was_updated = _update_pipeline_missing_preprocessing_parameters(curr_active_pipeline)
    print(f'were pipeline preprocessing parameters missing and updated?: {was_updated}')

    ## BUG 2023-05-25 - Found ERROR for a loaded pipeline where for some reason the filtered_contexts[long_epoch_name]'s actual context was the same as the short maze ('...maze2'). Unsure how this happened.
    long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
    long_epoch_context, short_epoch_context, global_epoch_context = [curr_active_pipeline.filtered_contexts[a_name] for a_name in (long_epoch_name, short_epoch_name, global_epoch_name)]
    # assert long_epoch_context.filter_name == long_epoch_name, f"long_epoch_context.filter_name: {long_epoch_context.filter_name} != long_epoch_name: {long_epoch_name}"
    # fix it if broken
    long_epoch_context.filter_name = long_epoch_name





def _on_complete_success_execution_session(curr_session_context, curr_session_basedir, curr_active_pipeline):
    """ called when the execute_session completes like:
        `post_run_callback_fn_output = post_run_callback_fn(curr_session_context, curr_session_basedir, curr_active_pipeline)`
        
        Meant to be assigned like:
        , post_run_callback_fn=_on_complete_success_execution_session
        
        Captures nothing.
        
        from Spike3D.scripts.run_BatchAnalysis import _on_complete_success_execution_session
        
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

    # ## Post Compute Validate 2023-05-16:
    post_compute_validate(curr_active_pipeline)
    
    ## Save the pipeline since that's disabled by default now:
    try:
        curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE) # AttributeError: 'PfND_TimeDependent' object has no attribute '_included_thresh_neurons_indx'
    except Exception as e:
        ## TODO: catch/log saving error and indicate that it isn't saved.
        print(f'ERROR SAVING PIPELINE for curr_session_context: {curr_session_context}. error: {e}')

    ## GLOBAL FUNCTION:
    # FIXME: doesn't seem like we should always use `force_recompute=True`
    try:
        # # 2023-01-* - Call extended computations to build `_display_short_long_firing_rate_index_comparison` figures:
        extended_computations_include_whitelist=['long_short_fr_indicies_analyses', 'jonathan_firing_rate_analysis', 'long_short_decoding_analyses', 'long_short_post_decoding'] # do only specifiedl
        newly_computed_values = batch_extended_computations(curr_active_pipeline, include_whitelist=extended_computations_include_whitelist, include_global_functions=True, fail_on_exception=True, progress_print=True, force_recompute=True, debug_print=False)
        print(f'newly_computed_values: {newly_computed_values}')        
        if len(newly_computed_values) > 0:
            print(f'newly_computed_values: {newly_computed_values}. Saving global results...')
            try:
                # Try to write out the global computation function results:
                curr_active_pipeline.save_global_computation_results()
            except Exception as e:
                print(f'!!WARNING!!: saving the global results threw the exception: {e}')
                print(f'\tthe global results are currently unsaved! proceed with caution and save as soon as you can!')
        else:
            print(f'no changes in global results.')
    except Exception as e:
        ## TODO: catch/log saving error and indicate that it isn't saved.
        print(f'ERROR SAVING GLOBAL COMPUTATION RESULTS for pipeline of curr_session_context: {curr_session_context}. error: {e}')
        

    # ### Programmatic Figure Outputs:
    try:
        neptuner = batch_perform_all_plots(curr_active_pipeline, enable_neptune=True, neptuner=None)
    except Exception as e:
        print(f'_perform_plots failed with exception: {e}')
        # raise e

    return {long_epoch_name:(long_laps, long_replays), short_epoch_name:(short_laps, short_replays),
            'outputs': {'local': curr_active_pipeline.pickle_path,
                        'global': curr_active_pipeline.global_computation_results_pickle_path}
        }
    


# ==================================================================================================================== #
# MAIN FUNCTION                                                                                                        #
# ==================================================================================================================== #
def main(active_global_batch_result_filename='global_batch_result.pkl', perform_execute=False, force_reload=True, enable_neptune=False, debug_print=True):
    """ Main Run Function
    from pyphoplacecellanalysis.General.Batch.runBatch import main, BatchRun, run_diba_batch, run_specific_batch

    """
    global_data_root_parent_path = find_first_extant_path([Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data')])
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
    global_batch_result_file_path = global_data_root_parent_path.joinpath(active_global_batch_result_filename).resolve()
    

    if enable_neptune:
        project = neptune.init_project()
        project["general/global_batch_result_filename"] = active_global_batch_result_filename
        project["general/global_data_root_parent_path"] = global_data_root_parent_path.as_posix()

    ## Currently contains an explicit neptune dependency:
    with neptune.init_run() as run:
        if enable_neptune:
            run['parameters/perform_execute'] = perform_execute
            run['parameters/global_batch_result_file_path'] = global_batch_result_file_path.as_posix()
            # project["general/data_analysis"].upload("data_analysis.ipynb")
            run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images"

        # Build `global_batch_run` pre-loading results (before execution)
        global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch

        if enable_neptune:
            # Pre-execution dataframe view:
            run["dataset/global_batch_run_progress_df"].upload(File.as_html(global_batch_run.to_dataframe(expand_context=True, good_only=False))) # "path/to/test_preds.csv"

        # Run Batch Executions/Computations
        if perform_execute:
            ## Execute the non-global functions with the custom arguments.
            active_computation_functions_name_whitelist=['_perform_baseline_placefield_computation',
                                                    # '_perform_time_dependent_placefield_computation',
                                                    '_perform_extended_statistics_computation',
                                                    '_perform_position_decoding_computation', 
                                                    '_perform_firing_rate_trends_computation',
                                                    '_perform_pf_find_ratemap_peaks_computation',
                                                    # '_perform_time_dependent_pf_sequential_surprise_computation'
                                                    '_perform_two_step_position_decoding_computation',
                                                    # '_perform_recursive_latent_placefield_decoding'
                                                ]

            # All Sessions:
            global_batch_run.execute_all(force_reload=force_reload, skip_extended_batch_computations=True, post_run_callback_fn=_on_complete_success_execution_session,
                                **{'computation_functions_name_whitelist': active_computation_functions_name_whitelist,
                                'active_session_computation_configs': None}) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)
            # 4m 39.8s


        # ## Single Session:
        # curr_sess_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
        # global_batch_run.reset_session(curr_sess_context) ## reset the context so it can be ran fresh.
        # global_batch_run.execute_session(session_context=curr_sess_context, force_reload=True, skip_extended_batch_computations=True,
        #                                               computation_functions_name_whitelist=active_computation_functions_name_whitelist, active_session_computation_configs=None) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

        # Save `global_batch_run` to file:
        saveData(global_batch_result_file_path, global_batch_run) # Update the global batch run dictionary
        if enable_neptune:
            run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images" # update file progress post-load
            # Post-execution dataframe view:
            run["dataset/global_batch_run_progress_df"].upload(File.as_html(global_batch_run.to_dataframe(expand_context=True, good_only=False))) # "path/to/test_preds.csv"
        

        # run.stop() # don't call run.stop() inside the run.

    if enable_neptune:
        ## POST Run
        project.stop()
    return global_batch_run, global_batch_result_file_path


if __name__ == "__main__":
    """ run main function to perform batch processing. """
    global_batch_run, finalized_loaded_global_batch_result_pickle_path = main(active_global_batch_result_filename='global_batch_result_new.pkl', perform_execute=True, debug_print=True)
