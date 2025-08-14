# %%
%config IPCompleter.use_jedi = False
%pdb off
%load_ext autoreload
%autoreload 3
import sys
from copy import deepcopy
from typing import List, Dict, Optional, Union, Callable
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
import tables as tb
from copy import deepcopy
from datetime import datetime, timedelta
from attrs import define, field, Factory

# required to enable non-blocking interaction:
%gui qt5

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path
from pyphocorehelpers.function_helpers import function_attributes
from pyphocorehelpers.exception_helpers import CapturedException

# Jupyter interactivity:
import ipywidgets as widgets
from IPython.display import display
from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import JupyterButtonRowWidget

# pyPhoPlaceCellAnalysis:
# NeuroPy (Diba Lab Python Repo) Loading
from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
from neuropy.core.session.Formats.Specific.RachelDataSessionFormat import RachelDataSessionFormat
from neuropy.core.session.Formats.Specific.HiroDataSessionFormat import HiroDataSessionFormatRegisteredClass
from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update

## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.session.Formats.BaseDataSessionFormats import find_local_session_paths
from neuropy.core import Epoch

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData, loadData
import pyphoplacecellanalysis.General.Batch.runBatch
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun, BatchResultDataframeAccessor, run_diba_batch, BatchComputationProcessOptions, BatchSessionCompletionHandler, SavingOptions
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme

from neuropy.core.user_annotations import UserAnnotationsManager
from pyphoplacecellanalysis.General.Batch.runBatch import SessionBatchProgress
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults, AcrossSessionTables, AcrossSessionsVisualizations

from pyphocorehelpers.Filesystem.path_helpers import set_posix_windows

from pyphocorehelpers.exception_helpers import CapturedException
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import InstantaneousFiringRatesDataframeAccessor
from pyphoplacecellanalysis.General.Batch.runBatch import PipelineCompletionResult, BatchSessionCompletionHandler

from pyphocorehelpers.Filesystem.metadata_helpers import FilesystemMetadata, get_file_metadata
from pyphocorehelpers.Filesystem.path_helpers import discover_data_files, generate_copydict, copy_movedict, copy_file, save_copydict_to_text_file, read_copydict_from_text_file, invert_filedict
from pyphoplacecellanalysis.General.Batch.runBatch import get_file_str_if_file_exists
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import check_output_h5_files, copy_files_in_filelist_to_dest
from pyphoplacecellanalysis.General.Batch.runBatch import ConcreteSessionFolder, BackupMethods

from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import batch_perform_all_plots, BatchPhoJonathanFiguresHelper
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PAPER_FIGURE_figure_1_add_replay_epoch_rasters, PAPER_FIGURE_figure_1_full, PAPER_FIGURE_figure_3, main_complete_figure_generations

from neuropy.core.neuron_identities import NeuronIdentityDataframeAccessor
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import build_merged_neuron_firing_rate_indicies
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.DirectionalPlacefieldGlobalComputationFunctions import DirectionalPlacefieldGlobalComputationFunctions, DirectionalLapsHelpers

# BATCH_DATE_TO_USE = '2023-10-20' # used for filenames throught the notebook
# BATCH_DATE_TO_USE = '2023-10-18_Apogee' # used for filenames throught the notebook
# BATCH_DATE_TO_USE = '2023-11-15_Lab' # used for filenames throught the notebook
BATCH_DATE_TO_USE = '2023-11-15_GL' # used for filenames throught the notebook

# %%
active_global_batch_result_filename=f'global_batch_result_{BATCH_DATE_TO_USE}.pkl'

debug_print = False
known_global_data_root_parent_paths = [Path(r'/home/halechr/cloud/turbo/Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data')] # , Path(r'/home/halechr/FastData'), Path(r'/nfs/turbo/umms-kdiba/Data'), Path(r'/home/halechr/turbo/Data'), Path(r'W:\Data'), Path(r'/home/halechr/cloud/turbo/Data')
global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
## Build Pickle Path:
global_batch_result_file_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve() # Use Default

# try to load an existing batch result:
global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename,
						skip_root_path_conversion=False, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch

batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
batch_progress_df.batch_results.build_all_columns()
good_only_batch_progress_df.batch_results.build_all_columns()
batch_progress_df
with pd.option_context('display.max_rows', 10, 'display.max_columns', None):  # more options can be specified also
    # display(batch_progress_df)
    # display(good_only_batch_progress_df)
    display(batch_progress_df)

# %% [markdown]
# # Run Batch Executions/Computations

# %%
# Hardcoded included_session_contexts:
included_session_contexts = [
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'), # prev completed
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'), # prev completed
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'), # prev completed
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'), # prev completed
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-08_21-16-25'),
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-09_22-24-40'),
    IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'),
    IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30'),
    IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-10_12-25-50'),
    IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-09_16-40-54'),
    IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='two',session_name='2006-4-10_12-58-3'),
    IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_17-46-44'), # prev completed
    IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-02_19-28-0'),
    IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='11-03_12-3-25'),
    IdentifyingContext(format_name='kdiba',animal='pin01',exper_name='one',session_name='fet11-01_12-58-54'), # prev completed
]

good_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, included_session_contexts)
good_session_concrete_folders


# from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import generate_batch_single_session_scripts

# ## Build Slurm Scripts:
# session_basedirs_dict: Dict[IdentifyingContext, Path] = {a_session_folder.context:a_session_folder.path for a_session_folder in good_session_concrete_folders}
# included_session_contexts, output_python_scripts, output_slurm_scripts = generate_batch_single_session_scripts(global_data_root_parent_path, session_batch_basedirs=session_basedirs_dict, included_session_contexts=included_session_contexts, output_directory=Path('output/generated_slurm_scripts/').resolve(), use_separate_run_directories=True, should_perform_figure_generation_to_file=False)
# display(output_python_scripts)

included_session_batch_progress_df = batch_progress_df[np.isin(batch_progress_df['context'].values, included_session_contexts)]
with pd.option_context('display.max_rows', 10, 'display.max_columns', None):  # more options can be specified also
    display(included_session_batch_progress_df)

# %% [markdown]
# # Execute Batch

# %%
# %pdb on

# multiprocessing_kwargs = dict(use_multiprocessing=False, num_processes=1)
multiprocessing_kwargs = dict(use_multiprocessing=True, num_processes=3)
  
# Whether to output figures:
should_perform_figure_generation_to_file=False
# should_perform_figure_generation_to_file=True

## Included Session Contexts:
# included_session_contexts = batch_progress_df[np.logical_and(batch_progress_df['has_user_replay_annotations'], batch_progress_df['is_ready'])]['context'].to_numpy().tolist()

# Only require sessions to have replay annotations:
# included_session_contexts = batch_progress_df[batch_progress_df['has_user_replay_annotations']]['context'].to_numpy().tolist()

# included_session_contexts = batch_progress_df['context'].to_numpy().tolist()[:4] # Only get the first 6
# Limit the contexts to run to the last N:
# included_session_contexts = included_session_contexts[3:5]

# included_session_contexts = [included_session_contexts[3]]

# ALL
included_session_contexts = included_session_contexts

# ## No filtering the sessions:
# included_session_contexts = None

if included_session_contexts is not None:
    print(f'len(included_session_contexts): {len(included_session_contexts)}')
else:
    print(f'included_session_contexts is None so all session contexts will be included.')

# included_session_contexts

# No recomputing at all:
result_handler = BatchSessionCompletionHandler(force_reload_all=False,
                                                session_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=False, should_save=SavingOptions.NEVER), # , override_file=
                                                global_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=False, should_save=SavingOptions.NEVER),
                                                should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, should_generate_all_plots=True, saving_mode=PipelineSavingScheme.SKIP_SAVING, force_global_recompute=False,
                                                **multiprocessing_kwargs)

# # No Reloading
# result_handler = BatchSessionCompletionHandler(force_reload_all=False,
#                                                 session_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED),
#                                                 global_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=True, should_save=SavingOptions.IF_CHANGED),
#                                                 should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, should_generate_all_plots=True, saving_mode=PipelineSavingScheme.SKIP_SAVING, force_global_recompute=False,
#                                                 **multiprocessing_kwargs)


# # Forced Reloading:
# result_handler = BatchSessionCompletionHandler(force_reload_all=True,
#                                                 session_computations_options=BatchComputationProcessOptions(should_load=False, should_compute=True, should_save=SavingOptions.ALWAYS),
#                                                 global_computations_options=BatchComputationProcessOptions(should_load=False, should_compute=True, should_save=SavingOptions.ALWAYS),
#                                                 should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE, force_global_recompute=True,
#                                                 **multiprocessing_kwargs)


active_post_run_callback_fn = result_handler.on_complete_success_execution_session
# active_post_run_callback_fn = _temp_on_complete_success_execution_session


# def a_test_completion_function(self, global_data_root_parent_path, curr_session_context, curr_session_basedir, curr_active_pipeline, across_session_results_extended_dict: dict) -> dict:
#     # print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#     print(f'a_test_completion_function(curr_session_context: {curr_session_context}, curr_session_basedir: {str(curr_session_basedir)}, ...,across_session_results_extended_dict: {across_session_results_extended_dict})')
#     long_epoch_name, short_epoch_name, global_epoch_name = curr_active_pipeline.find_LongShortGlobal_epoch_names()
#     # long_session, short_session, global_session = [curr_active_pipeline.filtered_sessions[an_epoch_name] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
#     # long_results, short_results, global_results = [curr_active_pip eline.computation_results[an_epoch_name]['computed_data'] for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
#     # Get existing laps from session:
#     # long_laps, short_laps, global_laps = [curr_active_pipeline.filtered_sessions[an_epoch_name].laps.as_epoch_obj() for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
#     # long_replays, short_replays, global_replays = [Epoch(curr_active_pipeline.filtered_sessions[an_epoch_name].replay.epochs.get_valid_df()) for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]
#     # long_PBEs, short_PBEs, global_PBEs = [curr_active_pipeline.filtered_sessions[an_epoch_name].pbe for an_epoch_name in [long_epoch_name, short_epoch_name, global_epoch_name]]


#     output_file_prefix = curr_session_context.get_description(separator="|", include_property_names=False)
#     print(f'-----------------_____------______---- BEGIN Directional Laps Result: \n\toutput_file_prefix: {output_file_prefix}\n\tcurr_active_pipeline.active_config_names: {curr_active_pipeline.active_config_names}')
#     directional_laps_result = curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']['computed_base_epoch_names']
#     print(f"\tsplit_directional_laps_names: {curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']['split_directional_laps_names']}")
#     print(f"\tsplit_directional_laps_names: {curr_active_pipeline.global_computation_results.computed_data['DirectionalLaps']['computed_base_epoch_names']}")
#     print(f'\n__ End Directional Laps result')

#     # jonathan_firing_rate_analysis_result = curr_active_pipeline.global_computation_results.computed_data.jonathan_firing_rate_analysis
#     # (epochs_df_L, epochs_df_S), (filter_epoch_spikes_df_L, filter_epoch_spikes_df_S), (good_example_epoch_indicies_L, good_example_epoch_indicies_S), (short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset), new_all_aclus_sort_indicies, assigning_epochs_obj = PAPER_FIGURE_figure_1_add_replay_epoch_rasters(curr_active_pipeline)
#     # neuron_replay_stats_df, short_exclusive, long_exclusive, BOTH_subset, EITHER_subset, XOR_subset, NEITHER_subset = jonathan_firing_rate_analysis_result.get_cell_track_partitions(frs_index_inclusion_magnitude=0.05)

#     # ## Output the BatchPhoJonathanFiguresHelper
#     # fig_1c_figures_all_dict = BatchPhoJonathanFiguresHelper.run(curr_active_pipeline, neuron_replay_stats_df.sort_values('custom_frs_index', ascending=True, inplace=False), included_unit_neuron_IDs=None,
# 	# n_max_page_rows=20, write_vector_format=False, write_png=True,
# 	# show_only_refined_cells=False, disable_top_row=False, split_by_short_long_shared=False)

    
#     # global_replays.filename = Path(f"output/{output_file_prefix}_global_replays").resolve()
#     # print(f'global_replays.filename: {global_replays.filename}')
#     # global_replays.to_neuroscope()

#     # global_PBEs.filename = Path(f"output/{output_file_prefix}_global_PBEs").resolve()
#     # print(f'global_PBEs.filename: {global_PBEs.filename}')
#     # global_PBEs.to_neuroscope('PBE')


#     # curr_active_pipeline, directional_lap_specific_configs = DirectionalLapsHelpers.split_to_directional_laps(curr_active_pipeline, add_created_configs_to_pipeline=True)
#     # curr_active_pipeline, directional_lap_specific_configs = constrain_to_laps(curr_active_pipeline)
#     # list(directional_lap_specific_configs.keys())

#     # joined_neruon_fri_df = build_merged_neuron_firing_rate_indicies(curr_active_pipeline, enable_display_intermediate_results=False)
#     # AcrossSessionTables.write_table_to_files(joined_neruon_fri_df, global_data_root_parent_path=global_data_root_parent_path, output_basename=f'{BATCH_DATE_TO_USE}_{output_file_prefix}_joined_neruon_fri_df')
#     print(f'>>\t done with {output_file_prefix}')
#     print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#     print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

#     return across_session_results_extended_dict


# result_handler.completion_functions.append(a_test_completion_function)

## Specific Setup for 2023-09-28 Changes to LxC/SxC "refinements"
result_handler.extended_computations_include_includelist = ['pf_computation', 'pfdt_computation', 'firing_rate_trends',
                                                'pf_dt_sequential_surprise',
                                                'ratemap_peaks_prominence2d',
                                                'position_decoding', 
                                                # 'position_decoding_two_step',
                                                # 'long_short_decoding_analyses',
                                                'jonathan_firing_rate_analysis', 'long_short_fr_indicies_analyses', 'short_long_pf_overlap_analyses', 'long_short_post_decoding', 'long_short_rate_remapping',
                                                # 'long_short_inst_spike_rate_groups',
                                                'long_short_endcap_analysis',
                                                'spike_burst_detection',
                                                'split_to_directional_laps',
                                                'rank_order_shuffle_analysis'
                                                ]


basic_local_computations = ['pf_computation', 'pfdt_computation', 'firing_rate_trends',
#                                                 'pf_dt_sequential_surprise',
                                                # 'ratemap_peaks_prominence2d',
                                                'position_decoding', 
                                                #'position_decoding_two_step', 
                                                ]
 
# result_handler.extended_computations_include_includelist = ['long_short_inst_spike_rate_groups']


result_handler.enable_hdf5_output = True # output the HDF5 when done.
# result_handler.override_existing_frs_index_values = True
# result_handler.frs_index_inclusion_magnitude = 0.1

# result_handler.enable_hdf5_output = False
result_handler.override_existing_frs_index_values = False


## Execute with the custom arguments.
global_batch_run.execute_all(force_reload=result_handler.force_reload_all, saving_mode=result_handler.saving_mode, skip_extended_batch_computations=True, post_run_callback_fn=active_post_run_callback_fn,
                             fail_on_exception=False, included_session_contexts=included_session_contexts,
                                                                                        **{'computation_functions_name_includelist': basic_local_computations,
                                                                                            'active_session_computation_configs': None,
                                                                                            'allow_processing_previously_completed': True}, **multiprocessing_kwargs) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

# 4m 39.8s

# %%
# Save to pickle:
saveData(global_batch_result_file_path, global_batch_run) # Update the global batch run dictionary

# Save to HDF5
suffix = f'{BATCH_DATE_TO_USE}'
## Build Pickle Path:
file_path = global_data_root_parent_path.joinpath(f'global_batch_output_{suffix}.h5').resolve()
global_batch_run.to_hdf(file_path,'/')

# %%
batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
batch_progress_df.batch_results.build_all_columns()
good_only_batch_progress_df.batch_results.build_all_columns()
good_only_batch_progress_df

# %% [markdown]
# # Across Sessions After Batching Complete

# %%
a_batch_progress_df = included_session_batch_progress_df.copy()

good_session_concrete_folders = [ConcreteSessionFolder(a_context, a_basedir) for a_context, a_basedir in zip(list(a_batch_progress_df.context.values), list(a_batch_progress_df.basedirs.values))]

# good_only_batch_progress_df.batch_results
# included_h5_paths = [get_file_str_if_file_exists(v.joinpath('output','pipeline_results.h5').resolve()) for v in list(good_only_batch_progress_df.basedirs.values)]
# included_h5_paths = [a_dir.joinpath('output','pipeline_results.h5').resolve() for a_dir in included_session_batch_progress_df['basedirs']]
included_h5_paths = [get_file_str_if_file_exists(v.pipeline_results_h5) for v in good_session_concrete_folders]

# %%
# target_dir = Path('output/across_session_results/2023-09-29').resolve()
# target_dir = Path('/home/halechr/cloud/turbo/Pho/Output/across_session_results/2023-09-29').resolve()
# target_dir = Path('/home/halechr/cloud/turbo/Pho/Output/across_session_results/2023-10-03').resolve()
# copy_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, target_dir=target_dir)
# copy_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, backup_mode=BackupMethods.RenameInSourceDirectory, rename_backup_suffix='2023-10-05', only_include_file_types=['local_pkl', 'global_pkl','h5'])
copy_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, backup_mode=BackupMethods.RenameInSourceDirectory, rename_backup_suffix=BATCH_DATE_TO_USE, only_include_file_types=['local_pkl', 'global_pkl'])
# copy_dict = ConcreteSessionFolder.build_backup_copydict(good_session_concrete_folders, backup_mode=BackupMethods.RenameInSourceDirectory, rename_backup_suffix='2023-10-07', only_include_file_types=['local_pkl', 'global_pkl','h5'])
copy_dict

# %%
moved_files_dict_h5_files = copy_movedict(copy_dict)
moved_files_dict_h5_files

# %%
moved_files_copydict_output_filename=f'backed_up_files_copydict_{BATCH_DATE_TO_USE}.csv'
moved_files_copydict_file_path = Path(global_data_root_parent_path).joinpath(moved_files_copydict_output_filename).resolve() # Use Default
print(f'moved_files_copydict_file_path: {moved_files_copydict_file_path}')

_out_string, filedict_out_path = save_copydict_to_text_file(moved_files_dict_h5_files, moved_files_copydict_file_path, debug_print=True)

# %%
read_moved_files_dict_files = read_copydict_from_text_file(moved_files_copydict_file_path, debug_print=False)
read_moved_files_dict_files

# %%
# read_moved_files_dict_files
restore_moved_files_dict_files = invert_filedict(read_moved_files_dict_files)
restore_moved_files_dict_files

# %%
check_output_h5_files(included_h5_paths)

# %% [markdown]
# ## Extract `across_sessions_instantaneous_fr_dict` from the computation outputs

# %%
# Somewhere in there there are `InstantaneousSpikeRateGroupsComputation` results to extract
across_sessions_instantaneous_fr_dict = {} # InstantaneousSpikeRateGroupsComputation
across_sessions_recomputed_instantaneous_fr_dict = {}

# Get only the sessions with non-None results
sessions_with_results = [a_ctxt for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None]
good_session_batch_outputs = {a_ctxt:a_result for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

for a_ctxt, a_result in good_session_batch_outputs.items():
    if a_result is not None:
        # a_good_result = a_result.__dict__.get('across_sessions_batch_results', {}).get('inst_fr_comps', None)
        a_good_result = a_result.across_session_results.get('inst_fr_comps', None)
        if a_good_result is not None:
            across_sessions_instantaneous_fr_dict[a_ctxt] = a_good_result
            # print(a_result['across_sessions_batch_results']['inst_fr_comps'])
        a_good_recomp_result = a_result.across_session_results.get('recomputed_inst_fr_comps', None)
        if a_good_recomp_result is not None:
            across_sessions_recomputed_instantaneous_fr_dict[a_ctxt] = a_good_recomp_result
            
num_sessions = len(across_sessions_instantaneous_fr_dict)
print(f'num_sessions: {num_sessions}')

# %%
across_sessions_recomputed_instantaneous_fr_dict

# %%
# When done, `result_handler.across_sessions_instantaneous_fr_dict` is now equivalent to what it would have been before. It can be saved using the normal `.save_across_sessions_data(...)`

## Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation)
# AcrossSessionsResults.save_across_sessions_data(across_sessions_instantaneous_fr_dict=across_sessions_instantaneous_fr_dict, global_data_root_parent_path=global_data_root_parent_path,
#                                                  inst_fr_output_filename=f'across_session_result_long_short_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl')


AcrossSessionsResults.save_across_sessions_data(across_sessions_instantaneous_fr_dict=across_sessions_recomputed_instantaneous_fr_dict, global_data_root_parent_path=global_data_root_parent_path,
                                                 inst_fr_output_filename=f'across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl')



# ## Save pickle:
# inst_fr_output_filename=f'across_session_result_long_short_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl'
# global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
# print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
# # Save the all sessions instantaneous firing rate dict to the path:
# saveData(global_batch_result_inst_fr_file_path, across_sessions_instantaneous_fr_dict)

# %%
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables

# neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=True, )

neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_and_save_all_combined_tables(included_session_contexts, included_h5_paths, override_output_parent_path=global_data_root_parent_path, output_path_suffix=f'{BATCH_DATE_TO_USE}')

# %%
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables

neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.build_all_known_tables(included_session_contexts, included_h5_paths, should_restore_native_column_types=True)
# neuron_replay_stats_table['is_refined_LxC']

# %%
long_short_fr_indicies_analysis_table

# %%
neuron_replay_stats_table

# %%
neuron_identities_table

# %%
# np.sum(neuron_replay_stats_table['is_refined_LxC'])
# np.isnan(neuron_replay_stats_table['is_refined_LxC'])

# %%
# Options
session_identifier_key: str = 'session_name'
# session_identifier_key: str = 'session_datetime'

## !IMPORTANT! Count of the fields of interest using .value_counts(...) and converting to an explicit pd.DataFrame:
# _out_value_counts_df: pd.DataFrame = neuron_replay_stats_table.value_counts(subset=['format_name', 'animal', 'session_name', 'session_datetime','track_membership'], normalize=False, sort=False, ascending=True, dropna=True).reset_index()
# _out_value_counts_df.columns = ['format_name', 'animal', 'session_name', 'session_datetime', 'track_membership', 'count']
_out_value_counts_df: pd.DataFrame = neuron_replay_stats_table.value_counts(subset=['format_name', 'animal', 'session_name', 'session_datetime','track_membership','is_refined_LxC', 'is_refined_SxC'], normalize=False, sort=False, ascending=True, dropna=True).reset_index()
_out_value_counts_df.columns = ['format_name', 'animal', 'session_name', 'session_datetime', 'track_membership', 'is_refined_LxC', 'is_refined_SxC', 'count']
_out_value_counts_df

# %%
## Find the time of the first session for each animal:
first_session_time  = _out_value_counts_df.groupby(['animal']).agg(session_datetime_first=('session_datetime', 'first')).reset_index()

## Subtract this initial time from all of the 'session_datetime' entries for each animal:
# Merge the first session time back into the original DataFrame
merged_df = pd.merge(_out_value_counts_df, first_session_time, on='animal')

# Subtract this initial time from all of the 'session_datetime' entries for each animal
merged_df['time_since_first_session'] = merged_df['session_datetime'] - merged_df['session_datetime_first']

merged_df

# %%
import matplotlib.pyplot as plt

point_size = 8
df = _out_value_counts_df.copy()
animals = df['animal'].unique()
track_memberships = df['track_membership'].unique()

fig, axes = plt.subplots(1, len(animals), figsize=(15, 5))

for i, animal in enumerate(animals):
	ax = axes[i]
	subset_df = df[df['animal'] == animal]
	
	for track_membership in track_memberships:
		track_subset_df = subset_df[subset_df['track_membership'] == track_membership]
		ax.plot(track_subset_df['session_datetime'], track_subset_df['count'], label=f'Track: {track_membership}')
		ax.scatter(track_subset_df['session_datetime'], track_subset_df['count'], s=point_size)
		
	ax.set_title(f'Animal: {animal}')
	ax.set_xlabel('Session Datetime')
	ax.set_ylabel('Count')
	ax.legend()

plt.tight_layout()
plt.show()

# %%
_out_value_counts_df

# %%


## See if the number of cells decreases over re-exposures to the track
df = _out_value_counts_df[_out_value_counts_df['animal'] == 'gor01']
# df = _out_value_counts_df[_out_value_counts_df['animal'] == 'pin01']
# df = _out_value_counts_df[_out_value_counts_df['animal'] == 'vvp01']

# Sort by column: 'session_datetime' (ascending)
df = df.sort_values(['session_datetime'])

'LEFT_ONLY'

# df.to_clipboard(index=False)
df

# %%
## Get the number of cells in each session of the animal:
num_LxCs = df[df['track_membership'] == 'LEFT_ONLY']['count'].to_numpy()
num_Shared = df[df['track_membership'] == 'SHARED']['count'].to_numpy()
num_SxCs = df[df['track_membership'] == 'RIGHT_ONLY']['count'].to_numpy()

num_TotalCs = num_LxCs + num_Shared + num_SxCs
num_TotalCs

# %%
# The only safe point to align each session to is the switchpoint (the delta):


# %%
# Each session can be expressed in terms of time from the start of the first session.


# %%
df.plot()


# %%

from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsVisualizations

matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
graphics_output_dict = AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions, save_figure=True)

# %% [markdown]
# ## Extract output files from all completed sessions:

# %%
from pyphocorehelpers.Filesystem.path_helpers import convert_filelist_to_new_parent

def save_filelist_to_text_file(hdf5_output_paths, filelist_path: Path):
    _out_string = '\n'.join([str(a_file) for a_file in hdf5_output_paths])
    print(f'{_out_string}')
    print(f'saving out to "{filelist_path}"...')
    with open(filelist_path, 'w') as f:
        f.write(_out_string)
    return _out_string, filelist_path

# Save output filelist:

# '/nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-09_1-22-43/output/pipeline_results.h5'

# kdiba_vvp01_two_2006-4-10_12-58-3
# 	outputs_local ={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/loadedSessPickle.pkl')}
# 	outputs_global ={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/output/global_computation_results.pkl'), 'hdf5': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/output/pipeline_results.h5')}
session_identifiers, pkl_output_paths, hdf5_output_paths = global_batch_run.build_output_files_lists()

h5_filelist_path = global_data_root_parent_path.joinpath(f'fileList_Greatlakes_HDF5_{BATCH_DATE_TO_USE}.txt').resolve()
_out_string, src_filelist_HDF5_savepath = save_filelist_to_text_file(hdf5_output_paths, h5_filelist_path)

pkls_filelist_path = global_data_root_parent_path.joinpath(f'fileList_Greatlakes_pkls_{BATCH_DATE_TO_USE}.txt').resolve()
_out_string, src_filelist_pkls_savepath = save_filelist_to_text_file(pkl_output_paths, pkls_filelist_path)

# source_parent_path = Path(r'/media/MAX/cloud/turbo/Data')
source_parent_path = Path(r'/nfs/turbo/umms-kdiba/Data')
dest_parent_path = Path(r'/~/W/Data/')
# # Build the destination filelist from the source_filelist and the two paths:
filelist_source = hdf5_output_paths
filelist_dest_paths = convert_filelist_to_new_parent(filelist_source, original_parent_path=source_parent_path, dest_parent_path=dest_parent_path)
filelist_dest_paths

dest_Apogee_h5_filelist_path = global_data_root_parent_path.joinpath(f'dest_fileList_Apogee_{BATCH_DATE_TO_USE}.txt').resolve()
_out_string, dest_filelist_savepath = save_filelist_to_text_file(filelist_dest_paths, dest_Apogee_h5_filelist_path)

# %%
from pyphoplacecellanalysis.General.Batch.runBatch import PipelineCompletionResult
from neuropy.core.epoch import Epoch

# Save to HDF5
suffix = f'{BATCH_DATE_TO_USE}'
## Build Pickle Path:
file_path = global_data_root_parent_path.joinpath(f'global_batch_output_{suffix}.h5').resolve()
file_path
global_batch_run.to_hdf(file_path,'/')

# %%
# Get only the sessions with non-None results
sessions_with_results = [a_ctxt for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None]

# list(global_batch_run.session_batch_outputs.keys())

# Somewhere in there there are `InstantaneousSpikeRateGroupsComputation` results to extract
across_sessions_instantaneous_fr_dict = {} # InstantaneousSpikeRateGroupsComputation

# good_session_batch_outputs = global_batch_run.session_batch_outputs

sessions_with_results = [a_ctxt for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None]
good_session_batch_outputs = {a_ctxt:a_result for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None}

for a_ctxt, a_result in good_session_batch_outputs.items():
    if a_result is not None:
        # a_good_result = a_result.__dict__.get('across_sessions_batch_results', {}).get('inst_fr_comps', None)
        a_good_result = a_result.across_session_results.get('inst_fr_comps', None)
        if a_good_result is not None:
            across_sessions_instantaneous_fr_dict[a_ctxt] = a_good_result
            # print(a_result['across_sessions_batch_results']['inst_fr_comps'])
            
num_sessions = len(across_sessions_instantaneous_fr_dict)
print(f'num_sessions: {num_sessions}')

# When done, `result_handler.across_sessions_instantaneous_fr_dict` is now equivalent to what it would have been before. It can be saved using the normal `.save_across_sessions_data(...)`

## Save the instantaneous firing rate results dict: (# Dict[IdentifyingContext] = InstantaneousSpikeRateGroupsComputation)
AcrossSessionsResults.save_across_sessions_data(across_sessions_instantaneous_fr_dict=across_sessions_instantaneous_fr_dict, global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=f'across_session_result_long_short_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl')

# ## Save pickle:
# inst_fr_output_filename=f'across_session_result_long_short_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl'
# global_batch_result_inst_fr_file_path = Path(global_data_root_parent_path).joinpath(inst_fr_output_filename).resolve() # Use Default
# print(f'global_batch_result_inst_fr_file_path: {global_batch_result_inst_fr_file_path}')
# # Save the all sessions instantaneous firing rate dict to the path:
# saveData(global_batch_result_inst_fr_file_path, across_sessions_instantaneous_fr_dict)

# %%
across_sessions_instantaneous_fr_dict

# %%
[a_ctxt.get_initialization_code_string() for a_ctxt in sessions_with_results]

# %% [markdown]
# # OLD

# %% [markdown]
# # 2023-10-06 - `joined_neruon_fri_df` loading

# %%
# BATCH_DATE_TO_USE = '2023-10-05_NewParameters'
BATCH_DATE_TO_USE = '2023-10-07'
all_sessions_joined_neruon_fri_df, out_path = build_and_merge_all_sessions_joined_neruon_fri_df(global_data_root_parent_path, BATCH_DATE_TO_USE)


# %%

joined_neruon_fri_df_basename = f'{BATCH_DATE_TO_USE}_{output_file_prefix}_joined_neruon_fri_df'
AcrossSessionTables.write_table_to_files(joined_neruon_fri_df, global_data_root_parent_path=global_data_root_parent_path, output_basename=joined_neruon_fri_df_basename, include_csv=False)
print(f'>>\t done with {output_file_prefix}')

# %%


# %% [markdown]
# # 2023-10-04 - Load Saved across-sessions-data and testing Batch-computed inst_firing_rates:

# %%
# from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update
# from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo, InstantaneousSpikeRateGroupsComputation
# from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends
# from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import list_of_dicts_to_dict_of_lists
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations


# %%
## Load the saved across-session results:
inst_fr_output_filename: str = f'across_session_result_long_short_recomputed_inst_firing_rate_{BATCH_DATE_TO_USE}.pkl'
across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
# across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
num_sessions = len(across_sessions_instantaneous_fr_dict)
print(f'num_sessions: {num_sessions}')

# %%
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionTables
 
## Load all across-session tables from the pickles:
output_path_suffix: str = f'{BATCH_DATE_TO_USE}'
neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table = AcrossSessionTables.load_all_combined_tables(override_output_parent_path=global_data_root_parent_path, output_path_suffix=output_path_suffix) # output_path_suffix=f'2023-10-04-GL-Recomp'
num_sessions = len(neuron_replay_stats_table.session_uid.unique().to_numpy())
print(f'num_sessions: {num_sessions}')

# %%
neuron_replay_stats_table

# %%
from neuropy.core.user_annotations import UserAnnotationsManager, SessionCellExclusivityRecord
from neuropy.utils.result_context import IdentifyingContext

annotation_man = UserAnnotationsManager()

LxC_uids = []
SxC_uids = []

for a_ctxt in included_session_contexts:
	session_uid = a_ctxt.get_description(separator="|", include_property_names=False)
	session_uid
	session_cell_exclusivity: SessionCellExclusivityRecord = annotation_man.annotations[a_ctxt].get('session_cell_exclusivity', None)
	LxC_uids.extend([f"{session_uid}|{aclu}" for aclu in session_cell_exclusivity.LxC])
	SxC_uids.extend([f"{session_uid}|{aclu}" for aclu in session_cell_exclusivity.SxC])
	
# [a_ctxt.get_description(separator="|", include_property_names=False) for a_ctxt in included_session_contexts]

long_short_fr_indicies_analysis_table['XxC_status'] = 'Shared'
long_short_fr_indicies_analysis_table.loc[np.isin(long_short_fr_indicies_analysis_table.neuron_uid, LxC_uids), 'XxC_status'] = 'LxC'
long_short_fr_indicies_analysis_table.loc[np.isin(long_short_fr_indicies_analysis_table.neuron_uid, SxC_uids), 'XxC_status'] = 'SxC'

## 2023-10-11 - Get the long peak location
long_short_fr_indicies_analysis_table['long_pf_peak_x'] = neuron_replay_stats_table['long_pf_peak_x']
# long_short_fr_indicies_analysis_table

# %%
matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
long_short_fr_indicies_analysis_table.plot.scatter(x='long_pf_peak_x', y='x_frs_index', title='Pf Peak position vs. LapsFRI', ylabel='Lap FRI')

long_short_fr_indicies_analysis_table.plot.scatter(x='long_pf_peak_x', y='y_frs_index', title='Pf Peak position vs. ReplayFRI', ylabel='Replay FRI')

# %% [markdown]
#  #TODO 2023-10-05 11:40: - [ ] Extract the "contrarian cells", the ones that have a strong exclusivity on the laps but the opposite tendency on the replays
# 

# %%
# long_short_fr_indicies_analysis_table_filename = 'output/2023-10-07_long_short_fr_indicies_analysis_table.csv'
long_short_fr_indicies_analysis_table_filename: str = 'output/{BATCH_DATE_TO_USE}_long_short_fr_indicies_analysis_table.csv'
long_short_fr_indicies_analysis_table.to_csv(long_short_fr_indicies_analysis_table_filename)
print(f'saved: {long_short_fr_indicies_analysis_table_filename}')

# %% [markdown]
# # 2023-10-10 - Statistics for `across_sessions_bar_graphs`, analysing `across_session_inst_fr_computation` 

# %%
import scipy.stats as stats
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import pho_stats_perform_diagonal_line_binomial_test, pho_stats_bar_graph_t_tests

binom_test_chance_result = pho_stats_perform_diagonal_line_binomial_test(long_short_fr_indicies_analysis_table)
print(f'binom_test_chance_result: {binom_test_chance_result}')

LxC_Laps_T_result, SxC_Laps_T_result, LxC_Replay_T_result, SxC_Replay_T_result = pho_stats_bar_graph_t_tests(across_session_inst_fr_computation)

# %% [markdown]
# ## 2023-10-04 - Run `AcrossSessionsVisualizations` corresponding to the PhoDibaPaper2023 figures for all sessions
# 

# %%
## Hacks the `PaperFigureTwo` and `InstantaneousSpikeRateGroupsComputation` 
global_multi_session_context, _out_aggregate_fig_2 = AcrossSessionsVisualizations.across_sessions_bar_graphs(across_session_inst_fr_computation, num_sessions, enable_tiny_point_labels=False, enable_hover_labels=False)


# %%
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionsVisualizations

matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
graphics_output_dict = AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions, save_figure=True)


# %%
matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
graphics_output_dict = AcrossSessionsVisualizations.across_sessions_long_and_short_firing_rate_replays_v_laps_figure(neuron_replay_stats_table=neuron_replay_stats_table, num_sessions=num_sessions, save_figure=True)


# %%
ann_man = UserAnnotationsManager()
included_annotations = {ctxt:ann_man.annotations[ctxt].get('session_cell_exclusivity', None) for ctxt in included_session_contexts}

all_LxCs = []
all_SxCs = []

for ctxt, an_ann in included_annotations.items():
	session_ctxt_key:str = ctxt.get_description(separator='|', subset_includelist=IdentifyingContext._get_session_context_keys())
	all_LxCs.extend([f"{session_ctxt_key}|{aclu}" for aclu in an_ann.LxC])
	all_SxCs.extend([f"{session_ctxt_key}|{aclu}" for aclu in an_ann.SxC])
	
all_LxCs

# %%
all_SxCs

# %%
across_session_inst_fr_computation.LxC_scatter_props
across_session_inst_fr_computation.SxC_scatter_props

# %%
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo
## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.

# To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).

## Display the aggregate across sessions:
_out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # WARNING: we didn't save this info
_out_fig_2.computation_result = across_session_inst_fr_computation # the result loaded from the file
_out_fig_2.active_identifying_session_ctx = across_session_inst_fr_computation.active_identifying_session_ctx
# Set callback, the only self-specific property
# _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)
# _out_fig_2.scatter_props_fn = _return_scatter_props_fn

# %%
LxC_aclus = _out_fig_2.computation_result.LxC_aclus
SxC_aclus = _out_fig_2.computation_result.SxC_aclus

LxC_aclus

# %%
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FigureOutputManager, FigureOutputLocation, ContextToPathMode

registered_output_files = {}

def output_figure(final_context: IdentifyingContext, fig, write_vector_format:bool=False, write_png:bool=True, debug_print=True):
    """ outputs the figure using the provided context. """
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_and_write_to_file
    def register_output_file(output_path, output_metadata=None):
        """ registers a new output file for the pipeline """
        print(f'register_output_file(output_path: {output_path}, ...)')
        registered_output_files[output_path] = output_metadata or {}

    fig_out_man = FigureOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.HIERARCHY_UNIQUE)
    active_out_figure_paths = build_and_write_to_file(fig, final_context, fig_out_man, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=register_output_file)
    return active_out_figure_paths, final_context


# Set callback, the only self-specific property
_out_fig_2._pipeline_file_callback_fn = output_figure

# %%
_out_fig_2.computation_result.Fig2_Laps_FR

# %%
_out_fig_2.computation_result.Fig2_Laps_FR

# %%
# Showing
restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
# Perform interactive Matplotlib operations with 'Qt5Agg' backend
_fig_2_theta_out, _fig_2_replay_out = _out_fig_2.display(active_context=global_multi_session_context, title_modifier_fn=lambda original_title: f"{original_title} ({num_sessions} sessions)", save_figure=True)
	
_out_fig_2.perform_save()

# %%
## 2023-10-11 - Surprise Shuffling


# %%



