# Greatlakes (GL) run instructions:

## Chained multi-phase submit (qclus12 pilot)

Generate **all** compute phases once, then submit per-session `afterok` chains so a finished session advances to the next phase without waiting for the cohort.

### Terminal 1: Generate scripts from `ProcessBatchOutputs_qclus1246789_Only.ipy`

```bash
cd '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D'
deactivate
source /scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate
export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
git pull
./scripts/unix/repos_pull_changes.sh
ipython ProcessBatchOutputs_qclus1246789_Only.ipy
```

## Lab Workstation:
```bash
cd '/home/halechr/repos/Spike3D'
deactivate
source /home/halechr/repos/Spike3D/.venv/bin/activate
export SPIKE3D_REPO_ROOT='/home/halechr/repos'
ipython ProcessBatchOutputs_qclus1246789_Only.ipy
```

```
cd '/home/halechr/repos/Spike3D'
deactivate
source /home/halechr/repos/Spike3D/.venv/bin/activate
export SPIKE3D_REPO_ROOT='/home/halechr/repos'
bash '/home/halechr/FastData/gen_scripts/submit_phase_chains_2026-10-28_GL.sh'
```

In `ProcessBatchOutputs_qclus12_Only.ipy`, set **`batch_execution_target`** before generating:

- **`'slurm'`** — Great Lakes (default for GL batch tags such as `_GL`)
- **`'linux_standalone'`** — Lab workstation (no `sbatch`; use `_Lab` batch tag)

This writes under `scripts_output_path` (GL: turbo `.../Output/gen_scripts/`; Lab: often `/home/halechr/FastData/gen_scripts/`):

- Per-session / per-phase run scripts + Python (`*_Clean`, `*_Continued`, `*_Final` suffixes)
- `submit_phase_chains_<BATCH_DATE_TO_USE>.sh` — **primary** submit path
- `submit_phase_chains_<BATCH_DATE_TO_USE>.json` — manifest (`execution_backend`: `slurm` or `bash`) of session → phase → script path
- `retry_phase_chains_<BATCH_DATE_TO_USE>/retry_from_<phase>_<session>.sh` — resume from a failed phase without re-running earlier phases

### Great Lakes: Terminal 2 (Slurm chains)

Set `batch_execution_target = 'slurm'`. Flat per-phase `sbatch '...'` lines are still printed as a fallback; prefer the chained submit script.

```bash
bash '/nfs/turbo/umms-kdiba/Data/Output/gen_scripts/submit_phase_chains_<BATCH_DATE_TO_USE>.sh'
# Example: bash '/nfs/turbo/umms-kdiba/Data/Output/gen_scripts/submit_phase_chains_2026-10-15_GL.sh'
```

Monitor:

```bash
squeue -u $USER
sacct -u $USER --starttime=now-2days --format=JobID,JobName%40,State,ExitCode,Elapsed -P
```

### Lab workstation: Terminal 2 (sequential bash, one session at a time)

Set `batch_execution_target = 'linux_standalone'` and a `_Lab` `BATCH_DATE_TO_USE`. The generator writes non-SLURM `run_*.sh` scripts and a master chain script that runs **one session at a time**, with **clean → continued → final** sequential `bash` inside each session (no `sbatch`).

```bash
bash '/home/halechr/FastData/gen_scripts/submit_phase_chains_<BATCH_DATE_TO_USE>.sh'
# Example: bash '/home/halechr/FastData/gen_scripts/submit_phase_chains_2026-10-20_Lab.sh'
```

Watch progress in the terminal, or tail log files next to each generated `run_*.sh`.

### On failure (do not re-run successful earlier phases)

Pickles from completed earlier phases stay on disk. A failed phase skips later phases for **that session only** (Slurm `afterok` on GL; `set -e` inside each session block on Lab).

**Great Lakes:** identify the failed phase with `sacct`, then:

```bash
bash '/nfs/turbo/umms-kdiba/Data/Output/gen_scripts/retry_phase_chains_<BATCH_DATE_TO_USE>/retry_from_continued_run_<session>.sh'
```

**Lab:** use the same retry script names under `retry_phase_chains_<BATCH_DATE_TO_USE>/`; they run sequential `bash` (no `sbatch`).

Example: if `continued_run` failed after a successful `clean_run`, use `retry_from_continued_run_...sh` (continued → final only). Do **not** re-submit clean.

### Notes

- Do **not** set `should_freeze_pipeline_updates=True` on compute phases (that skips compute/save). The abandoned Bapun / `test_all_script_phases` drivers had that bug.
- `ProcessBatchOutputs_qclus1246789_Only.ipy` still uses the manual single-phase flow until qclus12 chained mode is validated on GL; then port the same pattern.

---

## Legacy: single-phase generate (qclus1246789)

## Terminal 1: Generate the batch run scripts from `ProcessBatchOutputs_qclus1246789_Only.ipy`:
```bash
cd '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D'
deactivate
source /scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate
ipython ProcessBatchOutputs_qclus1246789_Only.ipy
```

## Terminal 2: Paste the flat `sbatch` run lines printed by the generator:
```bash
cd '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D'
deactivate
source /scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate

```


##



### Terminal: Run `Final`-phase Only:
```bash
cd '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D'
deactivate
source /scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate
export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
./scripts/unix/repos_pull_changes.sh
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_one_2006-6-08_14-26-15/run_kdiba_gor01_one_2006-6-08_14-26-15__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_one_2006-6-09_1-22-43/run_kdiba_gor01_one_2006-6-09_1-22-43__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_one_2006-6-12_15-55-31/run_kdiba_gor01_one_2006-6-12_15-55-31__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_two_2006-6-07_16-40-19/run_kdiba_gor01_two_2006-6-07_16-40-19__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_two_2006-6-08_21-16-25/run_kdiba_gor01_two_2006-6-08_21-16-25__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_two_2006-6-09_22-24-40/run_kdiba_gor01_two_2006-6-09_22-24-40__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_gor01_two_2006-6-12_16-53-46/run_kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_vvp01_two_2006-4-09_16-40-54/run_kdiba_vvp01_two_2006-4-09_16-40-54__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_vvp01_two_2006-4-10_12-58-3/run_kdiba_vvp01_two_2006-4-10_12-58-3__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_pin01_one_fet11-01_12-58-54/run_kdiba_pin01_one_fet11-01_12-58-54__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_pin01_one_11-02_17-46-44/run_kdiba_pin01_one_11-02_17-46-44__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
sbatch '/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts/run_kdiba_pin01_one_11-03_12-3-25/run_kdiba_pin01_one_11-03_12-3-25__withNormalComputedReplays-qclu_1246789-frateThresh_2.0_tbin_75ms_Final.sh'
```


# `IPython Cleanup Instructions

```bash
cd '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D'
deactivate
source /scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate
export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
git pull
./scripts/unix/repos_pull_changes.sh
ipython ## run ipython interactively
```
### IPython
```python
>>>
%load_ext autoreload
%autoreload 3

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers
import PostBatchCopyResults as pbcr

ctx = pbcr.PostBatchCopyContext.init_from_paths(
    batch_day_date='2026-11-04', host_tag='GL', result_date_to_use='2026-11-04',
    cutoff_date=datetime(2026, 11, 2),
    data_root=Path('/scratch/kdiba_root/kdiba99/halechr/Data'),
	scripts_output_path=Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts'),
	collected_outputs_path=Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/collected_outputs'),
    is_dryrun=True,
)
ctx = pbcr.configure_run_main(ctx)



copy_dict, _ = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
	global_data_root_parent_path=ctx.global_data_root_parent_path,
    BATCH_DATE_TO_USE=ctx.batch_date_to_use,
    cuttoff_date=datetime(2026, 11, 2),
    target_dir=ctx.collected_outputs_path,  # hypothetical dest; nothing copied if is_dry_run=True
    custom_file_globs_dict={
        'h5': '**/*.h5',
        'csv': '**/*.csv',
        'pkl': '**/*.pkl',
    },
    session_rel_search_dir='/',
    # session_rel_search_dir='output/',  # keeps raw session-root .mat/.xml out
    is_dry_run=True,
    debug_print=True,
)
copy_dict
# Group by session: keys are source Paths under each session's output/


## Copy Inst Fr Outputs to Collected Outputs:
### e.g. '/scratch/kdiba_root/kdiba99/halechr/Data/KDIBA/gor01/one/2006-6-09_1-22-43/output/2026-11-03_recomputed_inst_fr_comps_1000.0.pkl'
collected_outputs_day_date_folder: Path = ctx.collected_outputs_path.joinpath(str(ctx.batch_date_to_use))
collected_outputs_day_date_folder.mkdir(exist_ok=True)
print(f'collected_outputs_day_date_folder: "{collected_outputs_day_date_folder}"')
copy_dict, _ = AcrossSessionHelpers._copy_exported_files_from_session_folder_to_collected_outputs(
	global_data_root_parent_path=ctx.global_data_root_parent_path,
    BATCH_DATE_TO_USE=ctx.batch_date_to_use,
    cuttoff_date=datetime(2026, 11, 1),
    target_dir=collected_outputs_day_date_folder,  # hypothetical dest; nothing copied if is_dry_run=True
    custom_file_globs_dict={
        'pkl': '*_recomputed_inst_fr_comp*.pkl',
    },
    session_rel_search_dir='output/',
    # session_rel_search_dir='output/',  # keeps raw session-root .mat/.xml out
    is_dry_run=True,
    debug_print=False,
)
copy_dict


python PostBatchCopyResults.py --run-copy-sess-inst-fr --batch-day-date ... --result-date ... --host-tag GL ...

```

## Inst Fr Copy
```python
%load_ext autoreload
%autoreload 3

from datetime import datetime
from pathlib import Path
from collections import defaultdict
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import AcrossSessionHelpers
import PostBatchCopyResults as pbcr

## Copy Inst Fr → session-prefixed staging + merged across-session pickles (PostBatchCopyResults --run-copy-sess-inst-fr)
# Prerequisite: same ctx as above (init_from_paths + configure_run_main)
ctx: pbcr.PostBatchCopyContext = pbcr.PostBatchCopyContext.init_from_paths(
    batch_day_date='2026-11-03', host_tag='GL', result_date_to_use='2026-11-03',
    cutoff_date=datetime(2026, 10, 29),
    data_root=Path('/scratch/kdiba_root/kdiba99/halechr/Data'),
	scripts_output_path=Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts'),
	collected_outputs_path=Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/collected_outputs'),
    is_dryrun=False,
)

ctx = pbcr.configure_run_main(ctx)
ctx

## Preferred one-liner (batch folder + flat recomputed_inst_fr_comps/, no nested result-date):
# across_sessions_recomputed_instantaneous_fr_dict, moved_files_dict_files, paths = pbcr.run_copy_sess_inst_fr_data_to_across_session_pkl(ctx, full_custom_suffix=f"{ctx.override_custom_pickle_suffix}-tbin_16ms")

from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import copy_session_inst_fr_data_to_across_session_pkl

instantaneous_time_bin_size_seconds_list = [0.001, 1000.0]  # same default as PostBatchCopyResults.py
full_custom_suffix = f"{ctx.override_custom_pickle_suffix}-tbin_25ms"  # must match session output/*.pkl names

print(f'RESULT_DATE_TO_USE: {ctx.result_date_to_use!r}')
print(f'collected_outputs_path: "{ctx.collected_outputs_path}"')
print(f'global_data_root_parent_path: "{ctx.global_data_root_parent_path}"')
print(f'full_custom_suffix: {full_custom_suffix!r}')
print(f'is_dryrun: {ctx.is_dryrun}')

collected_outputs_day_date_folder: Path = ctx.collected_outputs_path.joinpath(str(ctx.batch_date_to_use))
collected_outputs_day_date_folder.mkdir(exist_ok=True)
print(f'collected_outputs_day_date_folder: "{collected_outputs_day_date_folder}"')

across_sessions_recomputed_instantaneous_fr_dict, moved_files_dict_files, (filelist_path, filedict_out_path) = copy_session_inst_fr_data_to_across_session_pkl(
    RESULT_DATE_TO_USE=ctx.result_date_to_use,
    collected_outputs_path=collected_outputs_day_date_folder,
    instantaneous_time_bin_size_seconds_list=instantaneous_time_bin_size_seconds_list,
    global_data_root_parent_path=ctx.global_data_root_parent_path,
    full_custom_suffix=full_custom_suffix,
    nested_result_date_staging=False,  # parent is already .../{batch_date_to_use}; staging = .../recomputed_inst_fr_comps/
    dry_run=False,
    debug_print=ctx.debug_print,
)

print(f'filelist_path: {filelist_path}')
print(f'filedict_out_path: {filedict_out_path}')
print(f'time bins in across dict: {list(across_sessions_recomputed_instantaneous_fr_dict.keys())}')
# Staging: {collected_outputs_day_date_folder}/recomputed_inst_fr_comps/
# Merged outputs (when dry_run=False), e.g.:
# {collected_outputs_day_date_folder}/across_session_result_long_short_recomputed_inst_firing_rate_2026-11-02_1000.0.pkl
across_sessions_recomputed_instantaneous_fr_dict


```

## Cleanup
```python
## Cleanup (mtime-based conditional delete). Differs from the export inventory above:
## - inventory/copy uses export-filename parsing + cuttoff_date
## - cleanup uses filesystem mtime vs clean_cutoff_date
## Prefer is_dryrun=True until the candidate list looks right; set False / --execute only to delete.
from collections import defaultdict

cutoff = datetime(2025, 5, 1) # Files with mtime BEFORE this are candidates for conditional patterns
ctx = pbcr.PostBatchCopyContext.init_from_paths(
    batch_day_date='2026-11-03', host_tag='GL', result_date_to_use='2026-11-03',
    cutoff_date=cutoff,
	# data_root=Path('/home/halechr/FastData/Data'),
	# data_root=Path('/media/halechr/MAX/Data'),
    # data_root=Path('/scratch/kdiba_root/kdiba99/halechr/Data'),
	# scripts_output_path=Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts'),
	# collected_outputs_path=Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/collected_outputs'),
    is_dryrun=True,  # list only; set False to actually unlink
)

ctx = pbcr.configure_run_main(ctx)
# custom_file_globs_dict replaces DEFAULT_CLEAN_CONDITIONAL_DELETE_PATTERNS (always-delete *.pkltmp / *.pkl.bak stay).
# Prefer session_rel_search_dir='output/' so session-root raw/mat files and loadedSessPickle are not scanned.
deleted_file_list, archive_folder = pbcr.run_clean_old_files(
    ctx,
    archive_collected_outputs=False,
    clean_cutoff_date=cutoff,
    custom_file_globs_dict={
        'h5': '*.h5',
        'csv': '*.csv',
        'pkl': '*.pkl',
    },
	session_rel_search_dir='output/',
    # session_rel_search_dir='output/',
)
deleted_file_list


by_session = defaultdict(list)
for p in deleted_file_list:
    # session folder is typically .../animal/epoch/session_name
    by_session[p.parent].append(p)

for folder, paths in sorted(by_session.items(), key=lambda x: str(x[0])):
    print(folder, len(paths))

# Only paths whose mtime is before cutoff (conditional logic already did this for custom globs;
# always-delete tmp/bak may still appear regardless of age)
old_only = [p for p in deleted_file_list if p.stat().st_mtime < cutoff.timestamp()]
old_only

```