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
from neuropy.utils.matplotlib_helpers import matplotlib_file_only

## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths

# from PendingNotebookCode import _perform_batch_plot, _build_batch_plot_kwargs
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_load_session, batch_extended_computations, batch_programmatic_figures, batch_extended_programmatic_figures
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme
from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun
from pyphoplacecellanalysis.General.Batch.runBatch import run_diba_batch
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import LongShortPipelineTests
# from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots

from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import main_complete_figure_generations


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
            active_computation_functions_name_includelist=['_perform_baseline_placefield_computation',
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
                                **{'computation_functions_name_includelist': active_computation_functions_name_includelist,
                                'active_session_computation_configs': None}) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)
            # 4m 39.8s


        # ## Single Session:
        # curr_sess_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
        # global_batch_run.reset_session(curr_sess_context) ## reset the context so it can be ran fresh.
        # global_batch_run.execute_session(session_context=curr_sess_context, force_reload=True, skip_extended_batch_computations=True,
        #                                               computation_functions_name_includelist=active_computation_functions_name_includelist, active_session_computation_configs=None) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

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
