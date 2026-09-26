---
name: MultiColor OOM tracing
overview: Add flushed stage-boundary debug prints (and light RSS snapshots) inside `_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay` so a GreatLakes OOM kill leaves a clear last checkpoint in the batch log.
todos:
  - id: trace-helper
    content: Add _trace(+optional RSS) helper and flush=True on existing prints in MultiColor display
    status: in_progress
  - id: stage-markers
    content: Insert stage 01–10 enter/exit traces with epoch counts around compute, deepcopy, exports, flat merge
    status: pending
  - id: export-progress
    content: Add progress_print mid-loop in export_decoded_posteriors_as_images; pass True from MultiColor exports
    status: pending
isProject: false
---

# MultiColor OOM stage tracing

## Context

GL job for `2006-6-09_1-22-43` was OOM-killed ~2 min after:

```
trying "_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay"
```

with no further flushed markers. Existing prints inside [`EpochComputationFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/EpochComputationFunctions.py) `_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay` (~L2723–3269) mostly lack `flush=True`, so buffered stdout can vanish on SIGKILL. Likely hotspots after entry: prerequisite recompute, `deepcopy` of PBE results, stage-1 `raw_rgba` export, stage-2 1D split + `greyscale`/`greyscale_shared_norm` export (scale with PBE count), then all-epoch `vertical_image_stack`.

## Approach

Instrument **only** `_display_decoded_trackID_weighted_position_posterior_withMultiColorOverlay` with a small local helper and stage markers. No memory-fix refactor in this pass.

### 1. Local `_trace` helper (inside the function)

```python
def _trace(msg: str):
    print(f'\t[MultiColorOverlay] {msg}', flush=True)
```

Optionally print process RSS when cheap (try `psutil.Process().memory_info().rss` → GB; if import fails, skip). Keep this behind the existing `debug_print` flag defaulting to True for this display (already `debug_print: bool = True`).

### 2. Stage checkpoints (print enter + exit / counts)

Add `_trace(...)` immediately **before** each heavy block, and again after with counts:

| Stage id | Where | Extra info |
|----------|-------|------------|
| `01_prereq_compute` | before/after `resolve_and_execute_full_required_computation_plan` (~L2842) | `force_recompute`, `time_bin_size` |
| `02_unpack_generic` | after loading `a_new_fully_generic_result` (~L2884) | |
| `03_build_laps_dict` | after building `decoder_laps_filter_epochs_decoder_result_dict` (~L2996) | `n_laps` per key |
| `04_build_pbe_dict` | after building `decoder_ripple_filter_epochs_decoder_result_dict` (~L3021) | `n_pbe` / `num_filter_epochs` per key |
| `05_filter_pbe` | after optional `filter_epochs_ripple_df` path (~L3044–3053) | filtered count or "no filter" |
| `06_stage1_raw_rgba_export` | before/after first `perform_export_all_decoded_posteriors_as_images` (~L3122) | formats=`raw_rgba`, `desired_height` |
| `07_split_1d` | before/after `split_pseudo2D_result_to_1D_result` loop (~L3141–3166) | which of laps/ripple enabled |
| `08_stage2_greyscale_export` | before/after second `perform_export_...` (~L3177) | formats=`greyscale`,`greyscale_shared_norm` |
| `09_flat_merge_stack` | before/after the all-epoch `vertical_image_stack` loop (~L3214) | n configs / n images |
| `10_done` | just before `return graphics_output_dict` | |

Also add `flush=True` to the existing nearby `print(...)` calls in this function so they survive hard kills.

### 3. Mid-export progress (minimal, same call chain)

In [`data_exporting.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/data_exporting.py) `export_decoded_posteriors_as_images` (~L599 loop), when `kwargs.get('progress_print', False)` (or every N epochs if `num_filter_epochs` is large):

```python
print(f'\t[PosteriorExport] epoch {i+1}/{num_filter_epochs} ...', flush=True)
```

Pass `progress_print=True` from the MultiColor display into both `perform_export_all_decoded_posteriors_as_images` calls so a kill mid-PBE-loop shows which epoch index died. Print every epoch only if `N <= 50`; else every `max(1, N // 20)` epochs plus first/last.

### 4. Out of scope

- Fixing OOM root cause (deepcopy / retaining all PIL images / giant vertical stacks)
- Slurm `--mem` / largemem changes
- Notebook edits

## Verification

Re-run (or wait for) a high-PBE GL session. On OOM, `*.log` should end on a flushed `[MultiColorOverlay] 0N_...` or `[PosteriorExport] epoch k/N` line, identifying the stage. On success, log shows the full 01→10 sequence.