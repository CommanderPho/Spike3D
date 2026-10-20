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


```bash
cd '/home/halechr/repos/Spike3D'
deactivate
source /home/halechr/repos/Spike3D/.venv/bin/activate
# Optional: export SPIKE3D_REPO_ROOT='/home/halechr/repos'
ipython ProcessBatchOutputs_qclus12_Only.ipy
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
