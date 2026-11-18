---
name: Ensure save parent dirs
overview: Fix `FileNotFoundError` when saving spikemats (and similar outputs) by ensuring the parent directory exists before any write. The minimal, robust approach is to centralize this in the low-level save helpers in `read_write.py`, plus the CSV path and the duplicate `save_data` in `forward_backward.py`.
todos:
  - id: read-write-ensure-dirs
    content: Add Path + _ensure_parent_dir; use in save_data, save_compressed_data, save_descriptive_stats
    status: completed
  - id: forward-backward-ensure
    content: Mirror parent mkdir in forward_backward.save_data before open
    status: completed
  - id: verify-preprocess
    content: Run preprocess_spikemat_data ripples + pytest if available
    status: in_progress
isProject: false
---

# Ensure output directories exist before saving

## Root cause

[`save_spikemat_data`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\read_write.py) builds paths like `DATA_PATH / str(data_type) / "*.obj"` (e.g. `.../data/ripples/rat1day1_4cm_3ms.obj`). The `ripples` subfolder is not guaranteed to exist in a fresh checkout. [`save_data`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\read_write.py) opens the file with no `mkdir`, so Python raises `FileNotFoundError`.

Dozens of `save_*` functions delegate to `save_data`, so fixing the bottom layer fixes ripple preprocessing, results under `RESULTS_PATH / <data_type> /`, and `DATA_PATH` subfolders in one shot.

## Implementation

**1. [`replay_structure/read_write.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\read_write.py)**

- Import `Path` from `pathlib` (already used elsewhere in the project via `DATA_PATH.joinpath`; this file currently does not import `Path`).
- Add a small helper, e.g. `_ensure_parent_dir(filename)`, that runs:

  `Path(filename).parent.mkdir(parents=True, exist_ok=True)`

- Call it at the start of **`save_data`** and **`save_compressed_data`** (both use `open(..., "wb")` the same way).
- Call it in **`save_descriptive_stats`** before `descriptive_stats.to_csv(filename)` — that path bypasses `save_data` today and would hit the same issue for missing `RESULTS_PATH/<data_type>/`.

No changes to individual `save_spikemat_data` / `save_gridsearch_results` / etc.; they all route through `save_data` except `to_csv`.

**2. [`replay_structure/forward_backward.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\HippocampalSWRDynamics\replay_structure\forward_backward.py)**

- This module defines its **own** `save_data` (lines 21–26) with the same pattern (pickle + `open` without mkdir). Add the same `Path(...).parent.mkdir(parents=True, exist_ok=True)` before the `open` call, with `from pathlib import Path`, to avoid a separate failure mode and to avoid importing the heavy `read_write` module into this file.

## Verification

- Re-run the failing command:  
  `python scripts/local/preprocess_spikemat_data.py --data_type ripples`  
  Confirm the `ripples` directory is created and `*.obj` writes succeed.
- Optionally run the existing test suite for HippocampalSWRDynamics (`uv run pytest` from that repo) to ensure nothing regresses.

## Scope note

This is intentionally limited to **write** paths. Load paths remain unchanged (missing files still error as before).
