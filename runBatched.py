# %%
from pathlib import Path
import pandas as pd

## Pho's Custom Libraries:
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path

# pyPhoPlaceCellAnalysis:
# NeuroPy (Diba Lab Python Repo) Loading

## For computation parameters:
from neuropy.utils.result_context import IdentifyingContext

from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
from pyphoplacecellanalysis.General.Batch.runBatch import BatchRun, BatchComputationProcessOptions, BatchSessionCompletionHandler
from pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline import PipelineSavingScheme


BATCH_DATE_TO_USE = '2023-08-08' # used for filenames throught the notebook


# %%
# active_global_batch_result_filename='global_batch_result_2023-07-12.pkl'
# active_global_batch_result_filename='global_batch_result_2023-07-07.pkl'
# active_global_batch_result_filename='global_batch_result_2023-07-07_extra.pkl'
# active_global_batch_result_filename='global_batch_result_2023-07-20.pkl'
# active_global_batch_result_filename='global_batch_result_2023-07-21_Greatlakes.pkl'
active_global_batch_result_filename=f'global_batch_result_{BATCH_DATE_TO_USE}.pkl'
debug_print = False
enable_neptune = False

if enable_neptune:
    import neptune # for logging progress and results
    from neptune.types import File
    from pyphoplacecellanalysis.General.Batch.NeptuneAiHelpers import set_environment_variables, neptune_output_figures
    neptune_kwargs = {'project':"commander.pho/PhoDibaLongShort2023",
    'api_token':"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOGIxODU2My1lZTNhLTQ2ZWMtOTkzNS02ZTRmNzM5YmNjNjIifQ=="}
    set_environment_variables(neptune_kwargs)
""" 
from pyphoplacecellanalysis.General.Batch.runBatch import main, BatchRun, run_diba_batch, run_specific_batch

"""
known_global_data_root_parent_paths = [Path(r'W:\Data'), Path(r'/media/MAX/Data'), Path(r'/Volumes/MoverNew/data'), Path(r'/home/halechr/turbo/Data'), Path(r'/nfs/turbo/umms-kdiba/Data')]
global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"

## NEPTUNE:
if enable_neptune:
	project = neptune.init_project(**neptune_kwargs)
	project["general/global_batch_result_filename"] = active_global_batch_result_filename
	project["general/global_data_root_parent_path"] = global_data_root_parent_path.as_posix()

## TODO: load the batch result initially:

## Build Pickle Path:
global_batch_result_file_path = Path(global_data_root_parent_path).joinpath(active_global_batch_result_filename).resolve() # Use Default

if enable_neptune:
	run = neptune.init_run() # rember to run.stop()
	run['parameters/global_batch_result_file_path'] = global_batch_result_file_path.as_posix()
	# project["general/data_analysis"].upload("data_analysis.ipynb")
	run["dataset/latest"].track_files(f"file://{global_batch_result_file_path}") # "s3://datasets/images"



# %%

# try to load an existing batch result:
# with set_posix_windows():
global_batch_run = BatchRun.try_init_from_file(global_data_root_parent_path, active_global_batch_result_filename=active_global_batch_result_filename,
						skip_root_path_conversion=False, debug_print=debug_print) # on_needs_create_callback_fn=run_diba_batch

batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
batch_progress_df.batch_results.build_all_columns()
good_only_batch_progress_df.batch_results.build_all_columns()
if enable_neptune:
	run["dataset/global_batch_run_progress_df"].upload(File.as_html(batch_progress_df)) # "path/to/test_preds.csv"
	run["dataset/global_batch_run_good_only_df"].upload(File.as_html(good_only_batch_progress_df)) # "path/to/test_preds.csv"

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    # display(batch_progress_df)
    # display(good_only_batch_progress_df)
    display(batch_progress_df)

# %% [markdown]
# # Run Batch Executions/Computations

# %%
# Hardcoded included_session_contexts:
included_session_contexts = [IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'),
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


# %%
# multiprocessing_kwargs = dict(use_multiprocessing=False, num_processes=1)
multiprocessing_kwargs = dict(use_multiprocessing=True, num_processes=8)

# Whether to output figures:
# should_perform_figure_generation_to_file=False
should_perform_figure_generation_to_file=True

## Included Session Contexts:
# included_session_contexts = batch_progress_df[np.logical_and(batch_progress_df['has_user_replay_annotations'], batch_progress_df['is_ready'])]['context'].to_numpy().tolist()

# Only require sessions to have replay annotations:
# included_session_contexts = batch_progress_df[batch_progress_df['has_user_replay_annotations']]['context'].to_numpy().tolist()

# included_session_contexts = batch_progress_df['context'].to_numpy().tolist()[:4] # Only get the first 6
## Limit the contexts to run to the last N:
# included_session_contexts = included_session_contexts[:2]

# ## No filtering the sessions:
# included_session_contexts = None

if included_session_contexts is not None:
    print(f'len(included_session_contexts): {len(included_session_contexts)}')
else:
    print(f'included_session_contexts is None so all session contexts will be included.')

# included_session_contexts


# No Reloading
result_handler = BatchSessionCompletionHandler(force_reload_all=False,
                                                session_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=False, should_save=False),
                                                global_computations_options=BatchComputationProcessOptions(should_load=True, should_compute=False, should_save=False),
                                                should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, saving_mode=PipelineSavingScheme.SKIP_SAVING, force_global_recompute=False,
                                                **multiprocessing_kwargs)

# # Forced Reloading:
# result_handler = BatchSessionCompletionHandler(force_reload_all=True,
#                                                 session_computations_options=BatchComputationProcessOptions(should_load=False, should_compute=True, should_save=True),
#                                                 global_computations_options=BatchComputationProcessOptions(should_load=False, should_compute=True, should_save=True),
#                                                 should_perform_figure_generation_to_file=should_perform_figure_generation_to_file, saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE, force_global_recompute=True,
#                                                 **multiprocessing_kwargs)



## Execute with the custom arguments.
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
# active_computation_functions_name_includelist=['_perform_baseline_placefield_computation']
global_batch_run.execute_all(force_reload=result_handler.force_reload_all, saving_mode=result_handler.saving_mode, skip_extended_batch_computations=True, post_run_callback_fn=result_handler.on_complete_success_execution_session,
                             fail_on_exception=False, included_session_contexts=included_session_contexts,
                                                                                        **{'computation_functions_name_includelist': active_computation_functions_name_includelist,
                                                                                            'active_session_computation_configs': None,
                                                                                            'allow_processing_previously_completed': True}, **multiprocessing_kwargs) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

# 4m 39.8s

# Get only the sessions with non-None results
sessions_with_results = [a_ctxt for a_ctxt, a_result in global_batch_run.session_batch_outputs.items() if a_result is not None]

# %%
# Save to pickle:
saveData(global_batch_result_file_path, global_batch_run) # Update the global batch run dictionary

# %%
# Combine across .h5 files:
import tables as tb

# f'/kdiba/gor01/one/2006-6-12_15-55-31/neuron_identities/table'

session_identifiers, pkl_output_paths, hdf5_output_paths

file_path = f'output/test_external_links.h5'
a_global_computations_group_key: str = f"{session_group_key}/global_computations"
with tb.open_file(file_path, mode='w') as f: # this mode='w' is correct because it should overwrite the previous file and not append to it.
    a_global_computations_group = f.create_group(session_group_key, 'global_computations', title='the result of computations that operate over many or all of the filters in the session.', createparents=True)
    an_external_link = f.create_external_link(f'', n

# %%


from typing import List
from attrs import define, field, Factory
import pandas as pd
import tables as tb

@define(slots=False)
class H5Loader:
    """
    H5Loader class for loading and consolidating .h5 files
    Usage:
        loader = H5Loader(file_list)
        master_table = loader.load_and_consolidate()
    """
    file_list: List[str] = field(default=Factory(list))
    table_key_list: List[str] = field(default=Factory(list))
    
    def load_and_consolidate(self) -> pd.DataFrame:
        """
        Loads .h5 files and consolidates into a master table
        """
        data_frames = []
        for file, table_key in zip(self.file_list, self.table_key_list):
            with tb.open_file(file) as h5_file:
                a_table = h5_file.get_node(table_key)
                print(f'a_table: {a_table}')
                # for a_record in a_table
                
                # data_frames.append(a_table)
#                 for table in h5_file.get_node(table_key):
#                 # for table in h5_file.root:
                # df = pd.DataFrame.from_records(a_table[:]) # .read()
                df = pd.DataFrame.from_records(a_table.read()) 
                data_frames.append(df)

        master_table = pd.concat(data_frames, ignore_index=True)
        return master_table

    
    
    
session_group_keys: List[str] = [("/" + a_ctxt.get_description(separator="/", include_property_names=False)) for a_ctxt in session_identifiers] # 'kdiba/gor01/one/2006-6-08_14-26-15'
# session_uid: str = session_context.get_description(separator="|", include_property_names=False)
neuron_identities_table_keys = [f"{session_group_key}/neuron_identities/table" for session_group_key in session_group_keys]


a_loader = H5Loader(file_list=hdf5_output_paths, table_key_list=neuron_identities_table_keys)
a_loader

_out_table = a_loader.load_and_consolidate()
_out_table

# %% [markdown]
# # Extract output files from all completed sessions:

# %%
# '/nfs/turbo/umms-kdiba/Data/KDIBA/gor01/one/2006-6-09_1-22-43/output/pipeline_results.h5'

# kdiba_vvp01_two_2006-4-10_12-58-3
# 	outputs_local ={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/loadedSessPickle.pkl')}
# 	outputs_global ={'pkl': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/output/global_computation_results.pkl'), 'hdf5': PosixPath('/nfs/turbo/umms-kdiba/Data/KDIBA/vvp01/two/2006-4-10_12-58-3/output/pipeline_results.h5')}

def build_output_files_lists(global_batch_run):
    # Uses `global_batch_run.session_batch_outputs`
    session_identifiers = []
    pkl_output_paths = []
    hdf5_output_paths = []

    for k, v in global_batch_run.session_batch_outputs.items():
        # v is PipelineCompletionResult
        if v is not None:
            # print(f'{k}')
            outputs_local = v.outputs_local
            outputs_global= v.outputs_global
            # print(f'\t{outputs_local =}\n\t{outputs_global =}\n\n')
            session_identifiers.append(k)
            if outputs_local.get('pkl', None) is not None:
                pkl_output_paths.append(outputs_local.get('pkl', None))
            if outputs_global.get('pkl', None) is not None:
                pkl_output_paths.append(outputs_global.get('pkl', None))
            if outputs_local.get('hdf5', None) is not None:
                hdf5_output_paths.append(outputs_local.get('hdf5', None))
            if outputs_global.get('hdf5', None) is not None:
                hdf5_output_paths.append(outputs_global.get('hdf5', None))

    return session_identifiers, pkl_output_paths, hdf5_output_paths

session_identifiers, pkl_output_paths, hdf5_output_paths = build_output_files_lists(global_batch_run)

# %%
session_identifiers

# %%
hdf5_output_paths

# %%


hdf5_output_paths

# %%
# Save output filelist:


def save_filelist_to_text_file(hdf5_output_paths, filelist_path: Path):
    _out_string = '\n'.join([str(a_file) for a_file in hdf5_output_paths])
    print(f'{_out_string}')
    print(f'saving out to "{filelist_path}"...')
    with open(filelist_path, 'w') as f:
        f.write(_out_string)
    return _out_string, filelist_path



h5_filelist_path = global_data_root_parent_path.joinpath(f'fileList_Greatlakes_HDF5_{BATCH_DATE_TO_USE}.txt').resolve()
_out_string, src_filelist_HDF5_savepath = save_filelist_to_text_file(hdf5_output_paths, h5_filelist_path)

pkls_filelist_path = global_data_root_parent_path.joinpath(f'fileList_Greatlakes_pkls_{BATCH_DATE_TO_USE}.txt').resolve()
_out_string, src_filelist_pkls_savepath = save_filelist_to_text_file(pkl_output_paths, pkls_filelist_path)


# %%
from pyphocorehelpers.Filesystem.path_helpers import convert_filelist_to_new_parent
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
dest_filelist_savepath

# %%
from pyphoplacecellanalysis.General.Batch.runBatch import PipelineCompletionResult
from neuropy.core.epoch import Epoch

# Save to HDF5
suffix = f'{BATCH_DATE_TO_USE}'
## Build Pickle Path:
file_path = global_data_root_parent_path.joinpath(f'global_batch_output_{suffix}.h5').resolve()
file_path

# %%
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

# %%
included_contexts = [IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'),
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


# %% [markdown]
# # 2023-07-14 - Load Saved across-sessions-data and testing Batch-computed inst_firing_rates:

# %%
from neuropy.utils.matplotlib_helpers import matplotlib_configuration_update
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import PaperFigureTwo
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.MultiContextComputationFunctions.LongShortTrackComputations import \
    InstantaneousSpikeRateGroupsComputation
from pyphoplacecellanalysis.General.Pipeline.Stages.ComputationFunctions.SpikeAnalysis import SpikeRateTrends
from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import list_of_dicts_to_dict_of_lists
from pyphoplacecellanalysis.General.Batch.AcrossSessionResults import AcrossSessionsResults, AcrossSessionsVisualizations

## Load the saved across-session results:
# inst_fr_output_filename = 'long_short_inst_firing_rate_result_handlers_2023-07-12.pkl'
# inst_fr_output_filename = 'across_session_result_long_short_inst_firing_rate.pkl'
# inst_fr_output_filename='across_session_result_long_short_inst_firing_rate_2023-07-21.pkl'
inst_fr_output_filename=f'across_session_result_handler_{BATCH_DATE_TO_USE}.pkl'
across_session_inst_fr_computation, across_sessions_instantaneous_fr_dict, across_sessions_instantaneous_frs_list = AcrossSessionsResults.load_across_sessions_data(global_data_root_parent_path=global_data_root_parent_path, inst_fr_output_filename=inst_fr_output_filename)
# across_sessions_instantaneous_fr_dict = loadData(global_batch_result_inst_fr_file_path)
num_sessions = len(across_sessions_instantaneous_fr_dict)
print(f'num_sessions: {num_sessions}')

# %%
## Hacks the `PaperFigureTwo` and `InstantaneousSpikeRateGroupsComputation` 
global_multi_session_context, _out_aggregate_fig_2 = AcrossSessionsVisualizations.across_sessions_bar_graphs(across_session_inst_fr_computation, num_sessions, enable_tiny_point_labels=False, enable_hover_labels=False)


# %%
across_session_inst_fr_computation.LxC_scatter_props
across_session_inst_fr_computation.SxC_scatter_props

# %%


# %%


# %%
# Build the unique scatter plot dictionaries:
across_session_contexts = list(across_sessions_instantaneous_fr_dict.keys())
unique_animals = IdentifyingContext.find_unique_values(across_session_contexts)['animal'] # {'gor01', 'pin01', 'vvp01'}
# Get number of animals to plot
marker_list = [(5, i) for i in np.arange(len(unique_animals))] # [(5, 0), (5, 1), (5, 2)]
scatter_props = [{'marker': mkr} for mkr in marker_list]  # Example, you should provide your own scatter properties
scatter_props_dict = dict(zip(unique_animals, scatter_props))
# {'pin01': {'marker': (5, 0)},
#  'gor01': {'marker': (5, 1)},
#  'vvp01': {'marker': (5, 2)}}
scatter_props_dict

# Pass a function that will return a set of kwargs for a given context
def _return_scatter_props_fn(ctxt: IdentifyingContext):
	""" captures `scatter_props_dict` """
	animal_id = str(ctxt.animal)
	return scatter_props_dict[animal_id]
	


# %%
## Aggregate across all of the sessions to build a new combined `InstantaneousSpikeRateGroupsComputation`, which can be used to plot the "PaperFigureTwo", bar plots for many sessions.
global_multi_session_context = IdentifyingContext(format_name='kdiba', num_sessions=num_sessions) # some global context across all of the sessions, not sure what to put here.

# To correctly aggregate results across sessions, it only makes sense to combine entries at the `.cell_agg_inst_fr_list` variable and lower (as the number of cells can be added across sessions, treated as unique for each session).

## Display the aggregate across sessions:
_out_fig_2 = PaperFigureTwo(instantaneous_time_bin_size_seconds=0.01) # WARNING: we didn't save this info
_out_fig_2.computation_result = across_session_inst_fr_computation # the result loaded from the file
_out_fig_2.active_identifying_session_ctx = across_session_inst_fr_computation.active_identifying_session_ctx
# Set callback, the only self-specific property
# _out_fig_2._pipeline_file_callback_fn = curr_active_pipeline.output_figure # lambda args, kwargs: self.write_to_file(args, kwargs, curr_active_pipeline)
_out_fig_2.scatter_props_fn = _return_scatter_props_fn

# %%
LxC_aclus = _out_fig_2.computation_result.LxC_aclus
SxC_aclus = _out_fig_2.computation_result.SxC_aclus

LxC_aclus

# %%
from pyphoplacecellanalysis.General.Mixins.ExportHelpers import FileOutputManager, FigureOutputLocation, ContextToPathMode

registered_output_files = {}

def output_figure(final_context: IdentifyingContext, fig, write_vector_format:bool=False, write_png:bool=True, debug_print=True):
    """ outputs the figure using the provided context. """
    from pyphoplacecellanalysis.General.Mixins.ExportHelpers import build_and_write_to_file
    def register_output_file(output_path, output_metadata=None):
        """ registers a new output file for the pipeline """
        print(f'register_output_file(output_path: {output_path}, ...)')
        registered_output_files[output_path] = output_metadata or {}

    fig_out_man = FileOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.HIERARCHY_UNIQUE)
    active_out_figure_paths = build_and_write_to_file(fig, final_context, fig_out_man, write_vector_format=write_vector_format, write_png=write_png, register_output_file_fn=register_output_file)
    return active_out_figure_paths, final_context


# Set callback, the only self-specific property
_out_fig_2._pipeline_file_callback_fn = output_figure

# %%
_out_fig_2.computation_result.Fig2_Laps_FR

# %%


# %%
_out_fig_2.computation_result.Fig2_Laps_FR

# %%
# Showing
restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
# Perform interactive Matplotlib operations with 'Qt5Agg' backend
_fig_2_theta_out, _fig_2_replay_out = _out_fig_2.display(active_context=global_multi_session_context, title_modifier_fn=lambda original_title: f"{original_title} ({num_sessions} sessions)", save_figure=True)
	
_out_fig_2.perform_save()

# %% [markdown]
# ## Single Session testing:
# 

# %%
_test_out = global_batch_run.execute_session(session_context=curr_sess_context, force_reload=True, skip_extended_batch_computations=True, computation_functions_name_includelist =['_perform_baseline_placefield_computation'], active_session_computation_configs=None) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)
_test_out

# global_batch_run.execute_session(session_context=curr_sess_context, force_reload=True, skip_extended_batch_computations=True, **{'computation_functions_name_includelist': ['_perform_baseline_placefield_computation'], 'active_session_computation_configs': None}) # can override `active_session_computation_configs` if we want to set custom ones like only the laps.)

# 23.5s

# %%

full_good_dirs = [k for k, v in global_batch_run.session_batch_errors.items() if v is None]
bad_dirs = [k for k, v in global_batch_run.session_batch_errors.items() if v is not None]
full_good_dirs
bad_dirs

# %%
global_batch_run.session_batch_status

# %%
global_batch_run.session_batch_status
global_batch_run.session_batch_errors

# %% [markdown]
# # Get good sessions for use in the specific session processing notebook:

# %%
batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=False) # all
good_only_batch_progress_df = global_batch_run.to_dataframe(expand_context=True, good_only=True)
good_only_batch_progress_df

# %%
## Get the list of sessions that are completely ready to process:
full_good_ready_to_process_sessions = list(good_only_batch_progress_df['context'].to_numpy())
full_good_ready_to_process_sessions
# Get good sessions for use in the specific session processing notebook:

# %%
run["good_sessions_list"].extend(full_good_ready_to_process_sessions)

# %%
run.stop()
project.stop()

# %%

print(",\n".join([ctx.get_initialization_code_string() for ctx in full_good_ready_to_process_sessions])) # List definitions

# [IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15'),
# IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43'),
# IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31'),
# IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6'),
# IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19'),
# IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46'),
# IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')]

# %%
print("\ncurr_context = ".join([ctx.get_initialization_code_string() for ctx in full_good_ready_to_process_sessions])) # Line definitions

# curr_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15')
# curr_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-09_1-22-43')
# curr_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-12_15-55-31')
# curr_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-13_14-42-6')
# curr_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-07_16-40-19')
# curr_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='two',session_name='2006-6-12_16-53-46')
# curr_context = IdentifyingContext(format_name='kdiba',animal='vvp01',exper_name='one',session_name='2006-4-09_17-29-30')

# %%
good_only_batch_progress_df

# %%
from datetime import datetime

# datetime object containing current date and time
save_time = datetime.now()
 
print("save_time =", save_time)

# dd/mm/YY H:M:S
dt_string = save_time.strftime("%Y-%m-%d_%I-%M%p")
print("date and time =", dt_string)

# %%
## Get output file paths:
completed_pipeline_filename = 'loadedSessPickle.pkl'
completed_global_computations_filename = 'outputs/global_computation_results.pkl'

full_good_ready_to_process_session_paths = list(good_only_batch_progress_df['basedirs'].to_numpy())
session_paths_output_folders = [sess_path.joinpath('outputs').resolve() for sess_path in full_good_ready_to_process_session_paths]

completed_pipeline_file_paths = [sess_path.joinpath(completed_pipeline_filename).resolve() for sess_path in full_good_ready_to_process_session_paths]
completed_global_computations_file_paths = [sess_path.joinpath(completed_global_computations_filename).resolve() for sess_path in full_good_ready_to_process_session_paths]
completed_global_computations_file_paths

# %%
# Countable Additivity 
# Any countable collections of points is size 0



# %%



