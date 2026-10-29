#!/usr/bin/env python3
"""PostBatchCopyResults — headless Great Lakes SSH script converted from ``PostBatchCopyResults.ipynb``.

Moves per-session batch outputs into shared ``collected_outputs`` / AcrossSession folders
without requiring a Jupyter kernel.

Run groups match the notebook cell tags (``run-main`` also includes the untagged copy/table pipeline):

  --run-main                 copy session H5 into collected_outputs, write copydict, build combined tables
  --run-copy-sess-inst-fr    merge session inst-FR exports into across-session pickles
  --run-clean-old-files      delete/archive old backup files in session data folders
  --all                      run all three groups (main, then inst-FR, then cleanup)

Usage (on remote, e.g. Great Lakes)::

    # Preview (default dry-run)
    python PostBatchCopyResults.py --run-main
    python PostBatchCopyResults.py --run-clean-old-files

    # Apply
    python PostBatchCopyResults.py --run-main --execute
    python PostBatchCopyResults.py --run-copy-sess-inst-fr --execute
    python PostBatchCopyResults.py --all --execute

    # Optional extras on --run-main
    python PostBatchCopyResults.py --run-main --execute --copy-figures --post-compute

    # Explicit data root / dates
    python PostBatchCopyResults.py --run-main --execute --data-root /nfs/turbo/umms-kdiba/Data --batch-day-date 2026-10-29 --host-tag GL --result-date 2026-10-20


    python PostBatchCopyResults.py --run-main --data-root /nfs/turbo/umms-kdiba/Data --batch-day-date 2026-10-29 --result-date 2026-10-29    

    --execute 

"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('MPLBACKEND', 'Agg')
if hasattr(os, 'getuid'):
    os.environ.setdefault('MPLCONFIGDIR', f'/tmp/mpl-cache-{os.getuid()}')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pandas as pd
from attrs import Factory, define, field

from neuropy.core.user_annotations import UserAnnotationsManager
from neuropy.utils.result_context import IdentifyingContext
from pyphocorehelpers.assertion_helpers import Assert
from pyphocorehelpers.Filesystem.path_helpers import copy_movedict, find_first_extant_path, save_copydict_to_text_file, save_filelist_to_text_file
from pyphoplacecellanalysis.General.Batch.runBatch import BackupMethods, ConcreteSessionFolder, get_file_path_if_file_exists
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers, AcrossSessionsResults, AcrossSessionTables, AcrossSessionsVisualizations, OldFileArchiver, check_output_h5_files, copy_batch_output_figures_to_common_figures_dir, copy_session_inst_fr_data_to_across_session_pkl


DEFAULT_BATCH_DAY_DATE: str = '2026-10-29'
DEFAULT_HOST_TAG: str = 'GL'
DEFAULT_RESULT_DATE: str = '2026-10-20'
DEFAULT_CLEAN_CUTOFF_DATE: str = '2025-02-20'

KNOWN_SCRIPTS_OUTPUT_PATHS: List[str] = [
    '/Users/pho/repo/Pho Secondary Workspace/Spike3DEnv/cloud/turbo/Data/Output/gen_scripts',
    '/home/halechr/cloud/turbo/Data/Output/gen_scripts/',
    '/Users/pho/University of Michigan Dropbox/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs',
    '/home/halechr/FastData/gen_scripts/',
    'output/gen_scripts/',
    'K:/scratch/gen_scripts',
]

KNOWN_COLLECTED_OUTPUT_PATHS: List[str] = [
    '/nfs/turbo/umms-kdiba/Data/Output/collected_outputs',
    '/home/halechr/FastData/collected_outputs/',
    '/home/halechr/cloud/turbo/Data/Output/collected_outputs',
    r'K:/scratch/collected_outputs',
    '/Users/pho/data/collected_outputs',
    'output/gen_scripts/',
    '/Volumes/SwapSSD/Data/collected_outputs',
    '/Users/pho/Dropbox (University of Michigan)/MED-DibaLabDropbox/Data/Pho/Outputs/output/collected_outputs',
    r'C:/Users/pho/repos/Spike3DWorkEnv/Spike3D/output/collected_outputs',
    '/home/halechr/FastData/collected_outputs/',
    '/home/halechr/cloud/turbo/Data/Output/collected_outputs',
]

KNOWN_GLOBAL_DATA_ROOT_PARENT_PATHS: List[str] = [
    '/home/halechr/cloud/turbo/Data',
    '/Users/pho/data',
    r'/nfs/turbo/umms-kdiba/Data',
    r'/media/halechr/BETAMAX/Data',
    r'/media/halechr/MAX/Data',
    r'W:/Data',
    r'/home/halechr/FastData',
    r'/Volumes/MoverNew/data',
]

NEURON_REPLAY_STATS_CSV_DTYPES: Dict[str, str] = {
    'long_pf_peak_x': 'float64',
    'short_pf_peak_x': 'float64',
    'long_LR_pf2D_peak_x': 'float64',
    'long_LR_pf2D_peak_y': 'float64',
    'long_RL_pf2D_peak_x': 'float64',
    'long_RL_pf2D_peak_y': 'float64',
    'short_LR_pf2D_peak_x': 'float64',
    'short_LR_pf2D_peak_y': 'float64',
    'short_RL_pf2D_peak_x': 'float64',
    'short_RL_pf2D_peak_y': 'float64',
    'long_LR_pf1D_peak': 'float64',
    'long_RL_pf1D_peak': 'float64',
    'short_LR_pf1D_peak': 'float64',
    'short_RL_pf1D_peak': 'float64',
    'peak_diff_RL_pf1D_peak': 'float64',
    'peak_diff_LR_pf1D_peak': 'float64',
}

DEFAULT_LOADED_TRACK_LIMITS: Dict[str, np.ndarray] = {
    'long_xlim': np.array([59.0774, 228.69]),
    'short_xlim': np.array([94.0156, 193.757]),
    'long_ylim': np.array([138.164, 146.12]),
    'short_ylim': np.array([138.021, 146.263]),
}


@define(slots=False)
class PostBatchCopyContext:
    """Resolved paths, sessions, and job-suffix config shared by every run group."""
    batch_day_date: str
    host_tag: str
    batch_date_to_use: str
    result_date_to_use: str
    cutoff_date: datetime
    collected_outputs_path: Path
    collected_figures_path: Path
    scripts_output_path: Path
    global_data_root_parent_path: Path
    included_session_contexts: List[IdentifyingContext]
    all_known_session_contexts: List[IdentifyingContext]
    good_session_concrete_folders: List[ConcreteSessionFolder]
    all_known_session_concrete_folders: List[ConcreteSessionFolder]
    session_basedirs_dict: Dict[IdentifyingContext, Path]
    job_suffix: str = ''
    override_custom_pickle_suffix: str = ''
    included_qclu_values: List[int] = Factory(list)
    minimum_inclusion_fr_Hz: float = 2.0
    laps_decoding_time_bin_size: float = 0.025
    ripple_decoding_time_bin_size: float = 0.025
    is_dryrun: bool = True
    debug_print: bool = False


def _parse_iso_date(value: str) -> datetime:
    return datetime.strptime(value, '%Y-%m-%d')


def _print_copy_dict_summary(copy_dict, label: str) -> None:
    n_items: int = len(copy_dict) if copy_dict is not None else 0
    print(f'{label}: {n_items} file(s)')
    if copy_dict is None:
        return
    for src, dest in list(copy_dict.items())[:20]:
        print(f'  {src}  ->  {dest}')
    ## END for src, dest in list(copy_dict.items())[:20]...
    if n_items > 20:
        print(f'  ... ({n_items - 20} more)')


def bootstrap_paths_and_sessions(batch_day_date: str, host_tag: str, result_date_to_use: str, cutoff_date: datetime, data_root: Optional[Path] = None, is_dryrun: bool = True, debug_print: bool = False) -> PostBatchCopyContext:
    """Resolve machine-specific paths and session lists (notebook cell 2)."""
    batch_date_to_use: str = f'{batch_day_date}_{host_tag}'

    known_scripts_output_paths = [Path(v).resolve() for v in KNOWN_SCRIPTS_OUTPUT_PATHS]
    scripts_output_path = find_first_extant_path(known_scripts_output_paths)
    assert scripts_output_path.exists(), f"scripts_output_path: {scripts_output_path} does not exist! Is the right computer's config commented out above?"
    print(f'scripts_output_path: {scripts_output_path}')

    known_collected_output_paths = [Path(v).resolve() for v in KNOWN_COLLECTED_OUTPUT_PATHS]
    known_collected_output_paths.append(scripts_output_path.joinpath('../collected_outputs').resolve())
    collected_outputs_path = find_first_extant_path(known_collected_output_paths)
    Assert.path_exists(collected_outputs_path)
    collected_outputs_path.mkdir(exist_ok=True)
    print(f'collected_outputs_path: {collected_outputs_path}')

    collected_figures_path = collected_outputs_path.parent.joinpath('collected_figures')
    collected_figures_path.mkdir(exist_ok=True)
    print(f'collected_figures_path: {collected_figures_path}')
    Assert.path_exists(collected_figures_path)

    all_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_good_sessions()
    excluded_session_keys: List[str] = []
    excluded_session_contexts = [IdentifyingContext.try_init_from_session_key(session_str=v) for v in excluded_session_keys]
    print(f'excluded_session_contexts: {excluded_session_contexts}')
    included_session_contexts: List[IdentifyingContext] = [v for v in all_session_contexts if v not in excluded_session_contexts]
    print(f'included_session_contexts: {included_session_contexts}')

    all_known_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_all_known_sessions()
    if data_root is not None:
        global_data_root_parent_path = data_root.resolve()
        if not global_data_root_parent_path.exists():
            raise FileNotFoundError(f'--data-root does not exist: {global_data_root_parent_path}')
    else:
        known_global_data_root_parent_paths = [Path(v) for v in KNOWN_GLOBAL_DATA_ROOT_PARENT_PATHS]
        global_data_root_parent_path = find_first_extant_path(known_global_data_root_parent_paths)
    assert global_data_root_parent_path.exists(), f"global_data_root_parent_path: {global_data_root_parent_path} does not exist! Is the right computer's config commented out above?"
    print(f'global_data_root_parent_path: {global_data_root_parent_path}')

    all_known_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, all_known_session_contexts)
    good_session_concrete_folders = ConcreteSessionFolder.build_concrete_session_folders(global_data_root_parent_path, included_session_contexts)
    session_basedirs_dict: Dict[IdentifyingContext, Path] = {a_session_folder.context: a_session_folder.path for a_session_folder in good_session_concrete_folders}

    return PostBatchCopyContext(
        batch_day_date=batch_day_date,
        host_tag=host_tag,
        batch_date_to_use=batch_date_to_use,
        result_date_to_use=result_date_to_use,
        cutoff_date=cutoff_date,
        collected_outputs_path=collected_outputs_path,
        collected_figures_path=collected_figures_path,
        scripts_output_path=scripts_output_path,
        global_data_root_parent_path=global_data_root_parent_path,
        included_session_contexts=included_session_contexts,
        all_known_session_contexts=all_known_session_contexts,
        good_session_concrete_folders=good_session_concrete_folders,
        all_known_session_concrete_folders=all_known_session_concrete_folders,
        session_basedirs_dict=session_basedirs_dict,
        is_dryrun=is_dryrun,
        debug_print=debug_print,
    )


def configure_run_main(ctx: PostBatchCopyContext) -> PostBatchCopyContext:
    """Active qclu / frate / tbin job suffix (notebook cell 3, tag ``run-main``)."""
    active_phase_dict = {}
    # ==================================================================================================================== #
    # Custom Suffix and overrides                                                                                          #
    # ==================================================================================================================== #
    # #'_withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]' ___________________________________________________________ #
    # active_phase_dict['override_custom_pickle_suffix'] = '_withNormalComputedReplays-qclu_[1, 2]-frateThresh_5.0'
    # included_qclu_values = [1,2]
    # minimum_inclusion_fr_Hz = 5.0
    # job_suffix = f"_withNormalComputedReplays-qclu_12-frateThresh_5.0" ## ends up being something like # job_suffix = f"_withNormalComputedReplays-qclu_12-frateThresh_5.0_tbin_25ms"

    # # # "_withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 9]" _______________________________________________ #
    # active_phase_dict['override_custom_pickle_suffix'] = "_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 9]-frateThresh_5.0" ##
    # included_qclu_values = [1, 2, 4, 6, 7, 9]
    # minimum_inclusion_fr_Hz = 5.0
    # job_suffix = f"_withNormalComputedReplays-qclu_124679-frateThresh_5.0"

    # # "_withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2]" _______________________________________________ #
    # active_phase_dict['override_custom_pickle_suffix'] = "_withNormalComputedReplays-qclu_[1, 2]-frateThresh_5.0"
    # included_qclu_values = [1, 2]
    # minimum_inclusion_fr_Hz = 5.0
    # job_suffix = f"_withNormalComputedReplays-qclu_12-frateThresh_5.0"

    # # # # "_withNormalComputedReplays-frateThresh_5.0-qclu_[1, 2, 4, 6, 7, 8, 9]" _______________________________________________ #
    # # active_phase_dict['override_custom_pickle_suffix'] = "_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_5.0"
    # included_qclu_values = [1, 2, 4, 6, 7, 8, 9]
    # minimum_inclusion_fr_Hz = 5.0
    # job_suffix = f"_withNormalComputedReplays-qclu_1246789-frateThresh_5.0"

    # # "_withNormalComputedReplays-frateThresh_2.0-qclu_[1, 2, 4, 6, 7, 8, 9]" _______________________________________________ #
    active_phase_dict['override_custom_pickle_suffix'] = "_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0"
    included_qclu_values = [1, 2, 4, 6, 7, 8, 9]
    minimum_inclusion_fr_Hz = 2.0
    job_suffix = f"_withNormalComputedReplays-qclu_1246789-frateThresh_2.0"

    # # # # "_withNormalComputedReplays-frateThresh_2.0-qclu_[1, 2, 4, 6, 7, 8, 9]" _______________________________________________ #
    # # active_phase_dict['override_custom_pickle_suffix'] = "_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_0.2"
    # included_qclu_values = [1, 2, 4, 6, 7, 8, 9]
    # minimum_inclusion_fr_Hz = 0.2
    # job_suffix = f"_withNormalComputedReplays-qclu_1246789-frateThresh_0.2"

    # ==================================================================================================================== #
    # Common:                                                                                                              #
    # ==================================================================================================================== #

    # laps_decoding_time_bin_size = None
    laps_decoding_time_bin_size = 0.025
    # laps_decoding_time_bin_size = 0.050
    # laps_decoding_time_bin_size = 0.075
    # laps_decoding_time_bin_size = 0.5

    ripple_decoding_time_bin_size = 0.025
    # ripple_decoding_time_bin_size = 0.050
    # ripple_decoding_time_bin_size = 0.058
    # ripple_decoding_time_bin_size = 0.075

    # job_suffix = f"{active_phase_dict['override_custom_pickle_suffix']}_tbin_25ms"
    # job_suffix = f"{job_suffix}_tbin_20ms"
    job_suffix = f"{job_suffix}_tbin_25ms"
    # job_suffix = f"{job_suffix}_tbin_50ms"
    # job_suffix = f"{job_suffix}_tbin_60ms"
    # job_suffix = f"{job_suffix}_tbin_75ms"

    # END ________________________________________________________________________________________________________________ #
    print(f'job_suffix: {job_suffix}')

    ctx.override_custom_pickle_suffix = active_phase_dict['override_custom_pickle_suffix']
    ctx.included_qclu_values = included_qclu_values
    ctx.minimum_inclusion_fr_Hz = minimum_inclusion_fr_Hz
    ctx.job_suffix = job_suffix
    ctx.laps_decoding_time_bin_size = laps_decoding_time_bin_size
    ctx.ripple_decoding_time_bin_size = ripple_decoding_time_bin_size
    return ctx


def run_main(ctx: PostBatchCopyContext, copy_figures: bool = False, post_compute: bool = False, plotting: bool = False, figures_export_date: Optional[str] = None) -> Dict[str, object]:
    """Copy session H5 into collected_outputs, write copydict, and build combined tables.

    Covers notebook cells 7–8, 17, 23, 33–34, 36, 39, 42, 45–46, plus optional figures (15) and post-compute (55–56).
    """
    def _subfn_across_session_h5_output_basename_fn_dated(session_context: Optional[IdentifyingContext], session_descr: Optional[str], basename: str, *args, separator_char: str = "_") -> str:
        """Builds output HDF5 file name for the dated collected_outputs subfolder (no BATCH_DATE_TO_USE prefix)."""
        if session_context is not None:
            session_descr = session_context.session_name  # '2006-6-07_16-40-19'
        _filename_list = [session_descr, basename]  # active_outputs_path is ./BATCH_DATE_TO_USE/ ... no BATCH_DATE_TO_USE prefix needed
        if len(args) > 0:
            _filename_list.extend([str(a_part) for a_part in args if a_part is not None])
        return separator_char.join(_filename_list)

    def _subfn_across_session_h5_output_basename_fn_prefixed(session_context: Optional[IdentifyingContext], session_descr: Optional[str], basename: str, *args, separator_char: str = "_") -> str:
        """Builds output HDF5 file name with BATCH_DATE_TO_USE prefix (collected_outputs root)."""
        if session_context is not None:
            session_descr = session_context.session_name  # '2006-6-07_16-40-19'
        _filename_list = [ctx.batch_date_to_use, session_descr, basename]
        if len(args) > 0:
            _filename_list.extend([str(a_part) for a_part in args if a_part is not None])
        return separator_char.join(_filename_list)

    def _subfn_normalize_str(s: str) -> str:
        return s.replace(' ', '').replace(',', '').replace('[', '').replace(']', '')

    def _subfn_pair_h5_paths_to_sessions(included_session_contexts: List[IdentifyingContext], h5_paths: List[Path]) -> Tuple[List[IdentifyingContext], List[Path]]:
        """Pair each session to an H5 whose filename contains ``session_name`` (replaces notebook .reverse() hack)."""
        paired_contexts: List[IdentifyingContext] = []
        paired_paths: List[Path] = []
        unmatched_paths: List[Path] = list(h5_paths)
        for a_ctxt in included_session_contexts:
            session_name: str = a_ctxt.session_name
            match: Optional[Path] = None
            for a_path in unmatched_paths:
                if session_name in a_path.name:
                    match = a_path
                    break
            ## END for a_path in unmatched_paths...
            if match is not None:
                paired_contexts.append(a_ctxt)
                paired_paths.append(match)
                unmatched_paths.remove(match)
            else:
                print(f'WARN: no H5 file matched session_name={session_name}')
        ## END for a_ctxt in included_session_contexts...
        if len(unmatched_paths) > 0:
            print(f'WARN: {len(unmatched_paths)} unmatched H5 path(s): {[p.name for p in unmatched_paths]}')
        return paired_contexts, paired_paths

    def _subfn_copy_session_h5_to_dated_collected() -> Tuple[dict, dict]:
        """Copy session ``*.h5`` into ``collected_outputs/{BATCH_DATE_TO_USE}/`` (cells 7–8)."""
        print(f'BATCH_DATE_TO_USE: "{ctx.batch_date_to_use}"')
        active_outputs_path = ctx.collected_outputs_path.joinpath(f'{ctx.batch_date_to_use}')
        active_outputs_path.mkdir(exist_ok=True)
        print(f'active_outputs_path: "{active_outputs_path.as_posix()}"')
        copy_dict, moved_files_dict_files = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
            BATCH_DATE_TO_USE=ctx.batch_date_to_use,
            cuttoff_date=ctx.cutoff_date,
            target_dir=active_outputs_path,
            custom_file_globs_dict={'h5': '*.h5'},
            is_dry_run=ctx.is_dryrun,
            rename_backup_basename_fn=_subfn_across_session_h5_output_basename_fn_dated,
            debug_print=ctx.debug_print,
        )
        _print_copy_dict_summary(copy_dict, 'dated collected_outputs H5 copy_dict')
        return copy_dict, moved_files_dict_files

    def _subfn_copy_parameterized_h5_and_write_manifest() -> Tuple[dict, Optional[dict], Path]:
        """Build parameterized ``pipeline_withNormalComputedReplays-...h5`` copydict and write CSV (cells 17, 23)."""
        included_h5_paths: List[Path] = []
        for v in ctx.good_session_concrete_folders:
            matching_files = list(v.output_folder.glob('pipeline_*.h5'))
            included_h5_paths.extend([p.resolve() for p in matching_files])
        ## END for v in ctx.good_session_concrete_folders...
        print(f'found {len(included_h5_paths)} session pipeline_*.h5 file(s)')
        if len(included_h5_paths) > 0:
            check_output_h5_files(included_h5_paths)

        parameterized_h5_name: str = f'pipeline{ctx.override_custom_pickle_suffix}.h5'
        custom_file_types_dict = {
            'custom_h5': (lambda a_session_folder, _name=parameterized_h5_name: get_file_path_if_file_exists(a_session_folder.output_folder.joinpath(_name).resolve())),
        }
        copy_h5_dict = ConcreteSessionFolder.build_backup_copydict(
            ctx.good_session_concrete_folders,
            target_dir=ctx.collected_outputs_path,
            backup_mode=BackupMethods.CommonTargetDirectory,
            rename_backup_basename_fn=_subfn_across_session_h5_output_basename_fn_prefixed,
            only_include_file_types=['h5'],
            custom_file_types_dict=custom_file_types_dict,
        )
        _print_copy_dict_summary(copy_h5_dict, 'parameterized pipeline H5 copy_h5_dict')

        h5_filelist_output_filename: str = f'{ctx.batch_date_to_use}_all_sessions_h5_filelist.txt'
        h5_filelist_output_file_path = Path(ctx.collected_outputs_path).joinpath(h5_filelist_output_filename).resolve()
        print(f'h5_filelist_output_file_path: {h5_filelist_output_file_path}')
        if not ctx.is_dryrun:
            save_filelist_to_text_file(included_h5_paths, filelist_path=h5_filelist_output_file_path, debug_print=True)
        else:
            print(f'DRY-RUN: would write h5 filelist to {h5_filelist_output_file_path}')

        moved_files_dict_h5_files = None
        moved_files_copydict_file_path = ctx.collected_outputs_path.joinpath(f'session_h5_files_copydict_{ctx.batch_date_to_use}.csv').resolve()
        print(f'moved_files_copydict_file_path: "{moved_files_copydict_file_path}"')
        if not ctx.is_dryrun:
            moved_files_dict_h5_files = copy_movedict(copy_h5_dict)
            _out_string, filedict_out_path = save_copydict_to_text_file(moved_files_dict_h5_files, moved_files_copydict_file_path, debug_print=True)
            print(f'filedict_out_path: "{filedict_out_path.as_posix()}"')
        else:
            print(f'DRY-RUN: would copy_movedict({len(copy_h5_dict)} items) and write {moved_files_copydict_file_path}')
        return copy_h5_dict, moved_files_dict_h5_files, moved_files_copydict_file_path

    def _subfn_collect_and_filter_collected_h5_paths() -> List[Path]:
        """Glob collected_outputs H5, filter by cutoff date and job suffix (cells 33–34, 36)."""
        glob_patterns = [
            f'{ctx.batch_date_to_use}_*_pipeline_with*.h5',
            f'{ctx.result_date_to_use}_{ctx.host_tag}_*_pipeline_with*.h5',
            '*_pipeline_with*.h5',
        ]
        matching_files: List[Path] = []
        for a_glob in glob_patterns:
            matching_files = list(ctx.collected_outputs_path.glob(a_glob))
            if len(matching_files) > 0:
                print(f'glob {a_glob!r} matched {len(matching_files)} file(s)')
                break
        ## END for a_glob in glob_patterns...
        if len(matching_files) == 0:
            print('WARN: no collected_outputs pipeline_with*.h5 files found')
            return []

        output_h5_files_df: pd.DataFrame = check_output_h5_files(matching_files)
        if (output_h5_files_df is None) or output_h5_files_df.empty:
            print('WARN: check_output_h5_files returned no rows')
            return [Path(p) for p in matching_files]

        output_h5_files_df['filename'] = output_h5_files_df['path'].map(lambda x: Path(x).name)
        output_h5_files_df = output_h5_files_df.sort_values(['creation_time', 'modification_time', 'file_size'], ascending=[False, False, False], inplace=False)
        if 'creation_time' in output_h5_files_df.columns:
            active_output_h5_files_df = output_h5_files_df[output_h5_files_df['creation_time'] >= ctx.cutoff_date]
        else:
            active_output_h5_files_df = output_h5_files_df
        active_included_h5_paths = active_output_h5_files_df['path'].map(lambda v: Path(v)).to_list()
        print(f'after cutoff_date={ctx.cutoff_date}: {len(active_included_h5_paths)} H5 path(s)')

        filename_normalized_job_suffix: str = _subfn_normalize_str(ctx.override_custom_pickle_suffix)
        initial_included_h5_paths: int = len(active_included_h5_paths)
        active_included_h5_paths = [a_path for a_path in active_included_h5_paths if (filename_normalized_job_suffix in _subfn_normalize_str(a_path.stem))]
        print(f'including: {len(active_included_h5_paths)}/{initial_included_h5_paths} h5 paths for override_custom_pickle_suffix: {ctx.override_custom_pickle_suffix}')
        return active_included_h5_paths

    def _subfn_build_and_save_combined_tables(paired_contexts: List[IdentifyingContext], paired_h5_paths: List[Path]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict]]:
        """``AcrossSessionTables.build_and_save_all_combined_tables`` plus CSV dtype re-read and remapping (cells 42, 45–46)."""
        from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import _add_cell_remapping_category
        assert (len(paired_contexts) == len(paired_h5_paths)), f'len(paired_contexts): {len(paired_contexts)} != len(paired_h5_paths): {len(paired_h5_paths)}'
        if len(paired_contexts) == 0:
            print('WARN: no paired session/H5 inputs; skipping combined tables')
            return None, None, None, None

        target_dir: Path = ctx.collected_outputs_path
        assert target_dir.exists(), f'target_dir: "{target_dir}" does not exist!'
        if ctx.is_dryrun:
            print(f'DRY-RUN: would build_and_save_all_combined_tables for {len(paired_contexts)} session(s) under {target_dir} / {ctx.batch_date_to_use}')
            for a_ctxt, a_path in zip(paired_contexts, paired_h5_paths):
                print(f'  {a_ctxt}  ->  {a_path}')
            ## END for a_ctxt, a_path in zip(paired_contexts, paired_h5_paths)...
            return None, None, None, None

        (neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_table), output_path_dicts = AcrossSessionTables.build_and_save_all_combined_tables(paired_contexts, paired_h5_paths, override_output_parent_path=target_dir, output_path_suffix=f'{ctx.batch_date_to_use}')
        curr_collected_outputs_folder = Path(output_path_dicts['neuron_replay_stats_table']['.csv']).resolve().parent
        print(f'combined tables folder: {curr_collected_outputs_folder}')

        _curr_read_path = output_path_dicts['neuron_replay_stats_table']['.csv']
        neuron_replay_stats_table = pd.read_csv(_curr_read_path, na_values=['', 'nan', 'np.nan', '<NA>'], dtype=NEURON_REPLAY_STATS_CSV_DTYPES, index_col='neuron_uid')
        neuron_replay_stats_df = deepcopy(neuron_replay_stats_table)
        neuron_replay_stats_df, (_non_disappearing_endcap_cells_df, _disappearing_endcap_cells_df, _minorly_changed_endcap_cells_df, significant_distant_remapping_endcap_cells_df) = _add_cell_remapping_category(neuron_replay_stats_df=neuron_replay_stats_df, loaded_track_limits=DEFAULT_LOADED_TRACK_LIMITS)
        print(f'significant_distant_remapping_endcap_cells_df: {len(significant_distant_remapping_endcap_cells_df)} row(s)')
        return neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_df, output_path_dicts

    def _subfn_copy_batch_figures() -> Optional[List[Path]]:
        """Copy gen_scripts batch figures into collected_figures (cell 15)."""
        export_date: str = figures_export_date if figures_export_date is not None else ctx.batch_day_date
        generate_figures_script_paths: List[Path] = []
        for a_ctxt in ctx.included_session_contexts:
            descr: str = a_ctxt.get_description(separator='_')
            candidate = ctx.scripts_output_path.joinpath(f'run_{descr}', f'figures_{descr}.py')
            if candidate.exists():
                generate_figures_script_paths.append(candidate)
            else:
                print(f'WARN: missing figures script: {candidate}')
        ## END for a_ctxt in ctx.included_session_contexts...
        if len(generate_figures_script_paths) == 0:
            generate_figures_script_paths = sorted(ctx.scripts_output_path.glob('run_*/figures_*.py'))
            print(f'fallback glob found {len(generate_figures_script_paths)} figures_*.py script(s)')
        if len(generate_figures_script_paths) == 0:
            print('WARN: no generate_figures_script_paths; skipping figure copy')
            return None
        _copied_outputs = copy_batch_output_figures_to_common_figures_dir(
            generate_figures_script_paths=generate_figures_script_paths,
            common_destination=ctx.collected_figures_path,
            previous_gen_scripts_path=ctx.scripts_output_path,
            curr_gen_scripts_path=None,
            batch_export_folder_date=export_date,
            is_dryrun=ctx.is_dryrun,
        )
        print(f'copied figures: {len(_copied_outputs) if _copied_outputs is not None else 0}')
        return _copied_outputs

    def _subfn_run_post_compute_and_plotting(long_short_fr_indicies_analysis_table: Optional[pd.DataFrame]) -> Optional[dict]:
        """Optional post-compute (cells 55–56) and Agg plotting (cell 51)."""
        graphics_output_dict = None
        if post_compute:
            if ctx.is_dryrun:
                print(f'DRY-RUN: would post_compute_all_sessions_processing(output_path_suffix={ctx.batch_date_to_use})')
            else:
                graphics_output_dict = AcrossSessionsResults.post_compute_all_sessions_processing(global_data_root_parent_path=ctx.collected_outputs_path, output_path_suffix=ctx.batch_date_to_use, plotting_enabled=plotting)
        if plotting and (long_short_fr_indicies_analysis_table is not None) and (not ctx.is_dryrun):
            num_sessions: int = int(long_short_fr_indicies_analysis_table.index.nunique()) if hasattr(long_short_fr_indicies_analysis_table.index, 'nunique') else len(ctx.included_session_contexts)
            graphics_output_dict = AcrossSessionsVisualizations.across_sessions_firing_rate_index_figure(long_short_fr_indicies_analysis_results=long_short_fr_indicies_analysis_table, num_sessions=num_sessions, save_figure=True)
        return graphics_output_dict

    print(f'BATCH_DATE_TO_USE: "{ctx.batch_date_to_use}"')
    print(f'collected_outputs_path: "{ctx.collected_outputs_path.as_posix()}"')
    dated_copy_dict, dated_moved = _subfn_copy_session_h5_to_dated_collected()
    copy_h5_dict, moved_files_dict_h5_files, copydict_path = _subfn_copy_parameterized_h5_and_write_manifest()
    active_included_h5_paths = _subfn_collect_and_filter_collected_h5_paths()
    paired_contexts, paired_h5_paths = _subfn_pair_h5_paths_to_sessions(ctx.included_session_contexts, active_included_h5_paths)
    print(f'paired {len(paired_contexts)} session(s) to {len(paired_h5_paths)} H5 file(s)')
    neuron_identities_table, long_short_fr_indicies_analysis_table, neuron_replay_stats_df, output_path_dicts = _subfn_build_and_save_combined_tables(paired_contexts, paired_h5_paths)
    copied_figures = _subfn_copy_batch_figures() if copy_figures else None
    graphics_output_dict = _subfn_run_post_compute_and_plotting(long_short_fr_indicies_analysis_table)
    return {
        'dated_copy_dict': dated_copy_dict,
        'dated_moved': dated_moved,
        'copy_h5_dict': copy_h5_dict,
        'moved_files_dict_h5_files': moved_files_dict_h5_files,
        'copydict_path': copydict_path,
        'paired_contexts': paired_contexts,
        'paired_h5_paths': paired_h5_paths,
        'neuron_identities_table': neuron_identities_table,
        'long_short_fr_indicies_analysis_table': long_short_fr_indicies_analysis_table,
        'neuron_replay_stats_df': neuron_replay_stats_df,
        'output_path_dicts': output_path_dicts,
        'copied_figures': copied_figures,
        'graphics_output_dict': graphics_output_dict,
    }


def run_copy_sess_inst_fr_data_to_across_session_pkl(ctx: PostBatchCopyContext, instantaneous_time_bin_size_seconds_list: Optional[List[float]] = None) -> Optional[Tuple]:
    """Merge session inst-FR exports into across-session pickles (notebook cell 27)."""
    if instantaneous_time_bin_size_seconds_list is None:
        instantaneous_time_bin_size_seconds_list = [0.001, 1000.0]
    result_date_to_use: str = ctx.result_date_to_use
    print(f'RESULT_DATE_TO_USE: {result_date_to_use}')
    print(f'instantaneous_time_bin_size_seconds_list: {instantaneous_time_bin_size_seconds_list}')
    if ctx.is_dryrun:
        print('DRY-RUN: would copy_session_inst_fr_data_to_across_session_pkl(...)')
        return None
    across_sessions_recomputed_instantaneous_fr_dict, moved_files_dict_files, (filelist_path, filedict_out_path) = copy_session_inst_fr_data_to_across_session_pkl(RESULT_DATE_TO_USE=result_date_to_use, collected_outputs_path=ctx.collected_outputs_path, instantaneous_time_bin_size_seconds_list=instantaneous_time_bin_size_seconds_list, debug_print=ctx.debug_print)
    print(f'filelist_path: {filelist_path}')
    print(f'filedict_out_path: {filedict_out_path}')
    n_keys: int = len(across_sessions_recomputed_instantaneous_fr_dict) if across_sessions_recomputed_instantaneous_fr_dict is not None else 0
    print(f'across_sessions_recomputed_instantaneous_fr_dict keys: {n_keys}')
    return across_sessions_recomputed_instantaneous_fr_dict, moved_files_dict_files, (filelist_path, filedict_out_path)


def run_clean_old_files(ctx: PostBatchCopyContext, archive_collected_outputs: bool = False, clean_cutoff_date: Optional[datetime] = None) -> Tuple[list, Optional[Path]]:
    """Delete old backup files in session folders (notebook cell 70); optionally archive collected_outputs (cell 72)."""
    always_delete_patterns = {
        '.pkltmp': '*.pkltmp',
        '.pkl.bak': '*.pkl.bak',
    }
    # always_delete_patterns = {} # disable non-conditional deletes while the pipeline is running in batch

    # Define patterns to conditionally delete
    conditional_delete_patterns = {
        # '.pkl': '*.pkl',
        'pipeline.pkl': '*loadedSessPickle*.pkl',
        'global.pkl': '*global_computation_results*.pkl',
        ## Numpy files:
        'position.npy': '*.position.npy',
        'mua.npy': '*.mua.npy',
        'pbe.npy': '*.pbe.npy',
        'non_pbe.npy': '*.non_pbe.npy',
        'interpolated_spike_positions.npy': '*.interpolated_spike_positions.npy',
        'non_pbe.npy': '*.non_pbe.npy',
    }

    ## Numpy files:
    numpy_files_patterns = {
        'position.npy': '*.position.npy',
        'mua.npy': '*.mua.npy',
        'pbe.npy': '*.pbe.npy',
        'non_pbe.npy': '*.non_pbe.npy',
        'interpolated_spike_positions.npy': '*.interpolated_spike_positions.npy',
        'non_pbe.npy': '*.non_pbe.npy',
        'ripple.npy': '*.ripple.npy',
    }

    mat_files_patterns = {
        'epochs_info.mat': '*.epochs_info.mat',
        'laps_info.mat': '*.laps_info.mat',
        'position_info.mat': '*.position_info.mat',
        'spikes.mat': '*.spikes.mat',
    }

    # always_delete_patterns = always_delete_patterns | numpy_files_patterns

    # Define the cutoff date
    # conditional_delete_files_cutoff_date = datetime(2023, 9, 1)  # Example cutoff date
    # conditional_delete_files_cutoff_date = datetime(2025, 8, 1)
    conditional_delete_files_cutoff_date = clean_cutoff_date if clean_cutoff_date is not None else datetime(2025, 2, 20)
    # conditional_delete_files_cutoff_date = datetime(2026, 10, 10)
    # conditional_delete_files_cutoff_date = None

    print(f'conditional_delete_files_cutoff_date: {conditional_delete_files_cutoff_date}')
    print(f'is_dryrun: {ctx.is_dryrun}')
    print(f'n_known_session_folders: {len(ctx.all_known_session_concrete_folders)}')
    deleted_file_list = OldFileArchiver.remove_backup_files_from_session_data_folders(
        good_session_concrete_folders=ctx.all_known_session_concrete_folders,
        always_delete_patterns=always_delete_patterns,
        conditional_delete_patterns=conditional_delete_patterns,
        conditional_delete_files_cutoff_date=conditional_delete_files_cutoff_date,
        is_dryrun=ctx.is_dryrun,
        debug_print=ctx.debug_print,
    )
    n_deleted: int = len(deleted_file_list) if deleted_file_list is not None else 0
    print(f'deleted_file_list: {n_deleted} file(s) {"would be removed" if ctx.is_dryrun else "removed"}')

    archive_folder: Optional[Path] = None
    if archive_collected_outputs:
        excluded_or_outdated_files_list: List[Path] = []
        print('archive_collected_outputs: no excluded_or_outdated_files_list was provided; creating OLD/ folder only')
        archive_folder = OldFileArchiver.archive_old_files(collected_outputs_directory=ctx.collected_outputs_path, excluded_or_outdated_files_list=excluded_or_outdated_files_list, is_dry_run=ctx.is_dryrun)
        print(f'archive_folder: {archive_folder}')
    return deleted_file_list, archive_folder


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Post-batch copy of session outputs into collected_outputs / AcrossSession folders (headless GL SSH).')
    run_group = parser.add_argument_group('run groups')
    run_group.add_argument('--run-main', action='store_true', help='Copy session H5, write copydict, build combined tables.')
    run_group.add_argument('--run-copy-sess-inst-fr', action='store_true', help='Merge session inst-FR exports into across-session pickles.')
    run_group.add_argument('--run-clean-old-files', action='store_true', help='Delete old backup files in session data folders.')
    run_group.add_argument('--all', action='store_true', help='Run all groups: main, then inst-FR, then cleanup.')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', default=True, help='Print planned copies/deletes only (default).')
    mode.add_argument('--execute', action='store_true', help='Actually copy / write / delete files.')

    parser.add_argument('--batch-day-date', type=str, default=DEFAULT_BATCH_DAY_DATE, help=f'Batch day date used in BATCH_DATE_TO_USE (default: {DEFAULT_BATCH_DAY_DATE}).')
    parser.add_argument('--host-tag', type=str, default=DEFAULT_HOST_TAG, help=f'Host tag appended to BATCH_DATE_TO_USE (default: {DEFAULT_HOST_TAG}).')
    parser.add_argument('--result-date', type=str, default=DEFAULT_RESULT_DATE, help=f'Result date for inst-FR / glob / cutoff default (default: {DEFAULT_RESULT_DATE}).')
    parser.add_argument('--cutoff-date', type=_parse_iso_date, default=None, help='H5 copy/glob cutoff date YYYY-MM-DD (default: --result-date).')
    parser.add_argument('--data-root', type=Path, default=None, help='Override data root (otherwise first extant known path).')
    parser.add_argument('--copy-figures', action='store_true', help='With --run-main, copy gen_scripts figures into collected_figures.')
    parser.add_argument('--figures-export-date', type=str, default=None, help='batch_export_folder_date for figure copy (default: --batch-day-date).')
    parser.add_argument('--post-compute', action='store_true', help='With --run-main, run AcrossSessionsResults.post_compute_all_sessions_processing.')
    parser.add_argument('--plotting', action='store_true', help='With --run-main, save Agg firing-rate-index figure.')
    parser.add_argument('--archive-collected-outputs', action='store_true', help='With --run-clean-old-files, ensure collected_outputs/OLD exists (no file list unless later supplied).')
    parser.add_argument('--clean-cutoff-date', type=_parse_iso_date, default=None, help=f'Conditional-delete cutoff for --run-clean-old-files (default: {DEFAULT_CLEAN_CUTOFF_DATE}).')
    parser.add_argument('--debug-print', action='store_true', help='Enable debug printing.')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    execute: bool = bool(args.execute)
    is_dryrun: bool = not execute

    run_main_flag: bool = bool(args.run_main or args.all)
    run_inst_fr_flag: bool = bool(args.run_copy_sess_inst_fr or args.all)
    run_clean_flag: bool = bool(args.run_clean_old_files or args.all)
    if not (run_main_flag or run_inst_fr_flag or run_clean_flag):
        print('No run group specified. Use --run-main, --run-copy-sess-inst-fr, --run-clean-old-files, or --all.')
        return 2

    cutoff_date: datetime = args.cutoff_date if args.cutoff_date is not None else _parse_iso_date(args.result_date)
    clean_cutoff_date: datetime = args.clean_cutoff_date if args.clean_cutoff_date is not None else _parse_iso_date(DEFAULT_CLEAN_CUTOFF_DATE)

    ctx = bootstrap_paths_and_sessions(
        batch_day_date=args.batch_day_date,
        host_tag=args.host_tag,
        result_date_to_use=args.result_date,
        cutoff_date=cutoff_date,
        data_root=args.data_root,
        is_dryrun=is_dryrun,
        debug_print=args.debug_print,
    )
    ctx = configure_run_main(ctx)

    print('-' * 80)
    print(f'BATCH_DATE_TO_USE: {ctx.batch_date_to_use}')
    print(f'result_date_to_use: {ctx.result_date_to_use}')
    print(f'cutoff_date: {ctx.cutoff_date}')
    print(f'job_suffix: {ctx.job_suffix}')
    print(f'n_included_sessions: {len(ctx.included_session_contexts)}')
    print(f'mode: {"EXECUTE" if execute else "DRY-RUN (no changes)"}')
    print(f'run_main: {run_main_flag}  run_copy_sess_inst_fr: {run_inst_fr_flag}  run_clean_old_files: {run_clean_flag}')
    print('-' * 80)

    if run_main_flag:
        print('[run-main]')
        run_main(ctx, copy_figures=args.copy_figures, post_compute=args.post_compute, plotting=args.plotting, figures_export_date=args.figures_export_date)
    if run_inst_fr_flag:
        print('[run-copy-sess-inst-fr]')
        run_copy_sess_inst_fr_data_to_across_session_pkl(ctx)
    if run_clean_flag:
        print('[run-clean-old-files]')
        run_clean_old_files(ctx, archive_collected_outputs=args.archive_collected_outputs, clean_cutoff_date=clean_cutoff_date)

    if not execute:
        print('\nRe-run with --execute to apply copies/deletes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
