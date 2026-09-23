---
name: Standalone pickle backup script
overview: Add a one-off standalone Python script (not the batch completion helper) that moves the current non-time-bin-suffixed qclus12 pipeline pickles to a `2026-09-23_75ms` suffix for the sessions in ProcessBatchOutputs_qclus12_Only.ipy, so they can be archived on GreatLakes before the 25ms clean_run.
todos:
  - id: add-script
    content: Create Spike3D/scripts/archive_qclus12_75ms_pickles.py with hardcoded qclus12 sessions, dry-run/execute flags, and shutil.move of session + global pickles (and h5 if present)
    status: completed
  - id: verify-paths
    content: Double-check stem/suffix naming matches loadedSessPickle{parameter_specifier}.pkl and global_computation_results{parameter_specifier}.pkl from the j2 template
    status: completed
isProject: false
---

# Standalone 75ms pickle archive script

## Goal

Create a **standalone `.py` script** you can scp/run on the remote (GL) to **move** (not copy) the current canonical qclus12 pickles aside with suffix `2026-09-23_75ms`, clearing the path for the upcoming 25ms `clean_run`.

**Do not change** [`backup_previous_session_files_completion_function`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) — that post-save hook is the wrong place for this.

## Chosen behavior

- **Operation:** `shutil.move` (archives old file and frees the canonical name)
- **Suffix:** `2026-09-23_75ms` (matches your current kwargs in the `.ipy`)
- **Pickle parameter suffix (unchanged):** `_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0`
- **Sessions:** the active list from [`ProcessBatchOutputs_qclus12_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus12_Only.ipy) (gor01 / vvp01 / pin01 sessions currently uncommented)
- **Safety:** default `--dry-run` prints planned moves; require `--execute` to actually move. Skip if source missing; **abort that file** if destination already exists (do not overwrite prior archives)

## Files moved per session

Under `{data_root}/KDIBA/{animal}/{exper_name}/{session_name}/`:

- `loadedSessPickle_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0.pkl`
  → `..._2026-09-23_75ms.pkl`
- `output/global_computation_results_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0.pkl`
  → `..._2026-09-23_75ms.pkl`

Also move `pipeline_results_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0.h5` if present (same stem suffix pattern).

## Script location and shape

New file: [`Spike3D/scripts/archive_qclus12_75ms_pickles.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\scripts\archive_qclus12_75ms_pickles.py)

- Stdlib-only preferred (`pathlib`, `shutil`, `argparse`) so it runs without loading Spike3D/pyphoplacecellanalysis
- Hardcode session relative paths as `KDIBA/{animal}/{exper}/{session}` strings matching the `.ipy` list
- Resolve data root via ordered candidates (same idea as the notebook): e.g. `/nfs/turbo/umms-kdiba/Data`, `/home/halechr/FastData`, `W:/Data`, plus `--data-root` override
- Print a clear summary: moved / skipped-missing / skipped-dest-exists / errors

## Remote usage

```bash
# Preview
python archive_qclus12_75ms_pickles.py --dry-run

# Apply
python archive_qclus12_75ms_pickles.py --execute
```

Then run the 25ms batch as usual; new pickles rewrite the cleared canonical names.

## Out of scope

- No edits to `batch_user_completion_helpers.py`
- No changes to pickle naming / `override_custom_pickle_suffix` in the batch notebook
- No SLURM wrapper (plain Python is enough)