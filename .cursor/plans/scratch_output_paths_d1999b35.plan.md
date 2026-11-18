---
name: Scratch Output Paths
overview: Prefer Great Lakes scratch under `/scratch/kdiba_root/kdiba99/halechr/Data/Output/` for NWB W-maze batch `gen_scripts` and `collected_outputs`, so figure completion pickles no longer write to full turbo.
todos:
  - id: update-path-list
    content: Preferscratch Data/Output/gen_scripts first (+ mkdir if Data exists) in NWB WMaze batch else-branch
    status: completed
  - id: verify-prints
    content: Confirm printed scripts/collected paths resolve under scratch after path setup
    status: completed
isProject: false
---

# Prefer Scratch Data/Output Over Turbo

## Problem

[`ProcessBatchOutputs_NWB_WMaze_Batch.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_NWB_WMaze_Batch.ipy) Slurm/default branch lists turbo first:

```324:331:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_NWB_WMaze_Batch.ipy
        known_scripts_output_paths = [Path(v).resolve() for v in ['/nfs/turbo/umms-kdiba/Pho/Output/gen_scripts', '/home/halechr/FastData/gen_scripts', '/home/halechr/cloud/turbo/Data/Output/gen_scripts', Path(r"W:/Data/Output/gen_scripts")]]
    scripts_output_path = find_first_extant_path(known_scripts_output_paths)
    ...
    collected_outputs_path = scripts_output_path.joinpath('../collected_outputs').resolve()
```

That bakes `/nfs/turbo/umms-kdiba/Pho/Output/collected_outputs` into generated figure scripts via `MAIN_get_template_string(..., collected_outputs_path=...)`, which then hits `[Errno 28] No space left on device` on turbo.

## Chosen approach

Prefer **`/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts`** (sibling `collected_outputs` under the same `Output/` parent). Matches existing Data preference for `/scratch/.../halechr/Data` on GL.

Scope: only the non-`linux_standalone` branch (Slurm / Windows / default). Leave `linux_standalone` FastData/BETAMAX ordering unchanged.

## Implementation (single file)

In [`ProcessBatchOutputs_NWB_WMaze_Batch.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_NWB_WMaze_Batch.ipy) ~lines 320–331:

1. Before `find_first_extant_path` on the else branch, if scratch Data root exists, `mkdir(parents=True, exist_ok=True)` for `/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts` so it can be selected even when newly created (otherwise turbo still wins).

2. Put that scratch path **first** in `known_scripts_output_paths`, keep turbo as fallback:

```python
else:
    scratch_scripts_output_path = Path('/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts')
    if Path('/scratch/kdiba_root/kdiba99/halechr/Data').exists():
        scratch_scripts_output_path.mkdir(parents=True, exist_ok=True)
    known_scripts_output_paths = [Path(v).resolve() for v in [scratch_scripts_output_path, '/nfs/turbo/umms-kdiba/Pho/Output/gen_scripts', '/home/halechr/FastData/gen_scripts', '/home/halechr/cloud/turbo/Data/Output/gen_scripts', Path(r"W:/Data/Output/gen_scripts")]]
```

No other logic changes: existing `collected_outputs_path = scripts_output_path.joinpath('../collected_outputs')` yields `/scratch/.../Data/Output/collected_outputs` when scratch is chosen.

## Verification

After edit / re-run of the batch driver path-setup section on GL:

- Printed `scripts_output_path` → `/scratch/kdiba_root/kdiba99/halechr/Data/Output/gen_scripts`
- Printed `collected_outputs_path` → `/scratch/kdiba_root/kdiba99/halechr/Data/Output/collected_outputs`
- Regenerated figure scripts embed the scratch `collected_outputs` path (already-generated turbo scripts still point at turbo until regenerated)

## Out of scope

- `ProcessBatchOutputs_Bapun_Batch.ipy` and other batch drivers
- Migrating existing turbo `collected_outputs` files
- Changing `linux_standalone` path order
