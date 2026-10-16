# Greatlakes (GL) run instructions:

## Chained multi-phase submit (qclus12 pilot)

Generate **all** compute phases once, then submit per-session `afterok` chains so a finished session advances to the next phase without waiting for the cohort.

### Terminal 1: Generate scripts from `ProcessBatchOutputs_qclus12_Only.ipy`

```bash
cd '/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D'
deactivate
source /scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv/Spike3D/.venv/bin/activate
# Optional: export SPIKE3D_REPO_ROOT='/scratch/kdiba_root/kdiba99/halechr/repos/Spike3D_ExploreEnv'
ipython ProcessBatchOutputs_qclus12_Only.ipy
```

This writes under `scripts_output_path` (typically turbo `.../Output/gen_scripts/`):

- Per-session / per-phase Slurm + Python scripts (`*_Clean`, `*_Continued`, `*_Final` suffixes)
- `submit_phase_chains_<BATCH_DATE_TO_USE>.sh` — **primary** submit path (per-session `sbatch --dependency=afterok:...` chains)
- `submit_phase_chains_<BATCH_DATE_TO_USE>.json` — manifest of session → phase → script path
- `retry_phase_chains_<BATCH_DATE_TO_USE>/retry_from_<phase>_<session>.sh` — resume from a failed phase without re-running earlier phases

Flat per-phase `sbatch '...'` lines are still printed as a fallback; prefer the chained submit script.

### Terminal 2: Submit the chains

```bash
bash '/nfs/turbo/umms-kdiba/Data/Output/gen_scripts/submit_phase_chains_<BATCH_DATE_TO_USE>.sh'
# Example: bash '/nfs/turbo/umms-kdiba/Data/Output/gen_scripts/submit_phase_chains_2026-10-15_GL.sh'
```

Monitor:

```bash
squeue -u $USER
sacct -u $USER --starttime=now-2days --format=JobID,JobName%40,State,ExitCode,Elapsed -P
```

### On failure (do not re-run successful earlier phases)

`afterok` cancels only that session’s later phases. Pickles from completed earlier phases stay on disk.

1. Identify the failed phase with `sacct`.
2. Re-submit from that phase onward, e.g.:

```bash
bash '/nfs/turbo/umms-kdiba/Data/Output/gen_scripts/retry_phase_chains_<BATCH_DATE_TO_USE>/retry_from_continued_run_<session>.sh'
```

Example: if `continued_run` failed after a successful `clean_run`, use `retry_from_continued_run_...sh` (requeues continued → final). Do **not** re-submit clean.

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
