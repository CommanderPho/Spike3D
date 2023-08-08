from pathlib import Path
import pathlib
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path

# pyPhoPlaceCellAnalysis:
# NeuroPy (Diba Lab Python Repo) Loading
# from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
# from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
# from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
# from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
# from neuropy.core.session.Formats.Specific.HiroDataSessionFormat import HiroDataSessionFormatRegisteredClass

## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun, BatchResultDataframeAccessor, run_diba_batch, BatchComputationProcessOptions, BatchSessionCompletionHandler, SessionBatchProgress, main
from pyphoplacecellanalysis.General.Batch.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations
# from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows


if __name__ == "__main__":
	""" run main function to perform batch processing. """
	BATCH_DATE_TO_USE = '2023-08-08' # used for filenames throught the notebook
	active_global_batch_result_suffix:str = f"{BATCH_DATE_TO_USE}_Apogee"
	print(f'active_global_batch_result_suffix: {active_global_batch_result_suffix}')

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


	global_batch_run, result_handler, across_sessions_instantaneous_fr_dict, output_filenames_tuple = main(active_global_batch_result_suffix=active_global_batch_result_suffix, included_session_contexts=included_session_contexts,
																										num_processes=1, should_force_reload_all=False, should_perform_figure_generation_to_file=False, debug_print=False)

	batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
	good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
	# good_only_batch_progress_df
	batch_progress_df
