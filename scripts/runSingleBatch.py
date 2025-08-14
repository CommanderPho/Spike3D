import argparse
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

## Pho's Custom Libraries:
from neuropy.utils.result_context import IdentifyingContext
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun, BatchResultDataframeAccessor, run_diba_batch, \
    SessionBatchProgress, main
from pyphoplacecellanalysis.General.Batch.BatchJobCompletion.BatchCompletionHandler import \
    BatchSessionCompletionHandler, BatchComputationProcessOptions
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations

def run_main(active_result_suffix, num_processes, should_force_reload_all, should_perform_figure_generation_to_file, debug_print):
    """ run main function to perform batch processing. """
    
    print(f'active_result_suffix: {active_result_suffix}')

    included_session_contexts: Optional[List[IdentifyingContext]] = [IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'),
        IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'),
        IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'),
        IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54')]


    global_batch_run, result_handler, across_sessions_instantaneous_fr_dict, output_filenames_tuple = main(active_result_suffix=active_result_suffix, 
                                                                                                        included_session_contexts=included_session_contexts,
                                                                                                        num_processes=num_processes, 
                                                                                                        should_force_reload_all=should_force_reload_all, 
                                                                                                        should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, 
                                                                                                        debug_print=debug_print)

    batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
    good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
    # good_only_batch_progress_df
    batch_progress_df

if __name__ == "__main__":
    # BATCH_DATE_TO_USE = '2023-08-08' # used for filenames throught the notebook
    # active_result_suffix:str = f"{BATCH_DATE_TO_USE}_Apogee"

    """ Usage:
    
    python scripts/runSingleBatch.py --active_result_suffix "2023-08-08_bApogee" --num_processes 4 --should_force_reload_all --debug_print
    python scripts/runSingleBatch.py --active_result_suffix "2023-08-08_LNX00052" --num_processes 4 --should_force_reload_all --debug_print
    
    --should_perform_figure_generation_to_file
    --should_perform_figure_generation_to_file
    
    """
    parser = argparse.ArgumentParser(description='Perform batch processing.')
    parser.add_argument('--active_result_suffix', required=True, help='Suffix used for filenames throughout the notebook.')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of processes to use.')
    parser.add_argument('--should_force_reload_all', action='store_true', help='Force reload all data.')
    parser.add_argument('--should_perform_figure_generation_to_file', action='store_true', help='Perform figure generation to file.')
    parser.add_argument('--debug_print', action='store_true', help='Enable debug printing.')

    args = parser.parse_args()

    run_main(args.active_result_suffix, args.num_processes, args.should_force_reload_all, args.should_perform_figure_generation_to_file, args.debug_print)
