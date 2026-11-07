---
name: Fix epoch name hardcodes
overview: Add safe_resolve_PrimarySecondaryGlobal_epoch_names() and route hardcoded include_includelist[0]/[1]/[-1] long/short/global sites through it, with call-site notes that Primary/Secondary map to Long/Short for kdiba.
todos:
  - id: add-resolver
    content: Add safe_resolve_PrimarySecondaryGlobal_epoch_names() on ComputedPipelineStage (+ pipeline wrapper if duplicated)
    status: completed
  - id: fix-overlap-comp
    content: Fix _perform_long_short_pf_overlap_analyses to use the helper (note Primary/Secondary = Long/Short for kdiba)
    status: completed
  - id: fix-displays
    content: Fix pf1D comparison, scalar overlap, rank-order z-stats, and Jonathan displays to use the helper with the same kdiba Long/Short note
    status: completed
  - id: fix-batch-global
    content: Replace included_computation_filter_names[-1] with find_Global_epoch_name() in batch helpers
    status: completed
isProject: false
---

# Fix hardcoded long/short/global epoch indexing

## Problem

With directional kdiba filters, `active_completed_computation_result_names` looks like:

`['maze1_odd', 'maze2_odd', 'maze_odd', 'maze1_even', ..., 'maze1_any', 'maze2_any', 'maze_any']`

Code that does `include_includelist[0]/[1]/[-1]` gets **odd** tracks for long/short, while jonathan stats and canonical PF membership use **`maze1_any` / `maze2_any`** via `find_LongShortGlobal_epoch_names()`.

## Approach

Add a shared pipeline helper next to the existing finders in [`Computation.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py) (~715–810). Name it for generality (Primary/Secondary/Global), not kdiba-specific Long/Short:

```python
def safe_resolve_PrimarySecondaryGlobal_epoch_names(self, include_includelist=None):
    """Resolve (primary, secondary, global) epoch filter names.

    For kdiba sessions: delegates to find_LongShortGlobal_epoch_names()
    so primary/secondary are the canonical Long/Short tracks (e.g. maze1_any / maze2_any),
    not directional odd/even prefixes from active_completed_computation_result_names.

    For non-kdiba sessions: falls back to include_includelist[0]/[1]/[-1]
    (or active_completed_computation_result_names when include_includelist is None).
    """
    if include_includelist is None:
        include_includelist = self.active_completed_computation_result_names
    if self.is_kdiba_session():
        return self.find_LongShortGlobal_epoch_names()  # Long, Short, Global
    primary_epoch_name = include_includelist[0]
    secondary_epoch_name = include_includelist[1]
    global_epoch_name = include_includelist[-1] if len(include_includelist) > 2 else ''
    return primary_epoch_name, secondary_epoch_name, global_epoch_name
```

Mirror on the pipeline wrapper class if `find_LongShortGlobal_epoch_names` / `is_kdiba_session` are duplicated there (~2989+).

**Call-site convention:** keep local variable names `long_epoch_name, short_epoch_name, global_epoch_name` where the surrounding code is Long/Short-specific, and add an explicit comment that for kdiba Primary/Secondary == Long/Short:

```python
# For kdiba: Primary/Secondary == Long/Short (maze1_any / maze2_any), not maze1_odd / maze2_odd.
long_epoch_name, short_epoch_name, global_epoch_name = owning_pipeline_reference.safe_resolve_PrimarySecondaryGlobal_epoch_names(include_includelist)
```

## Sites to update

### Computation (highest impact)

1. [`LongShortTrackComputations._perform_long_short_pf_overlap_analyses`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/LongShortTrackComputations.py) (~960–970) — currently unconditional `[0]/[1]/[-1]`. Sibling `_perform_jonathan_replay_firing_rate_analyses` already uses `find_LongShortGlobal_epoch_names()`. Wrong epochs here poison `short_long_pf_overlap_analyses` for all consumers.

### Display

2. `_display_long_short_pf1D_comparison` (~425–431) — uses long/short results for PF plots; still hardcoded.
3. `_display_short_long_pf1D_scalar_overlap_comparison` (~517–523) — epoch names mostly for debug print today, but align anyway; real data fix is (1).
4. `RankOrderGlobalDisplayFunctions._display_rank_order_z_stats_results` in [`RankOrderComputations.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/RankOrderComputations.py) (~3405–3411) — `short_epoch_name` / `global_epoch_name` used for `filtered_epochs` split times.
5. Replace the inline `is_kdiba_session()` blocks in the two Jonathan displays with the helper + kdiba Long/Short note.

### Batch global resolution

6. When `included_computation_filter_names` is explicit, prefer `find_Global_epoch_name()` over `[-1]` in:
   - [`NonInteractiveProcessing.batch_extended_computations`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py) (~507–508)
   - [`NonInteractiveProcessing.batch_evaluate_required_computations`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py) (~375; also ensure `global_epoch_name` is defined outside `progress_print`)
   - [`ComputedPipelineStage.batch_extended_computations`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py) (~1873)

(`[-1]` is often already `maze_any`, but `find_Global_epoch_name()` matches the `None` path and avoids depending on list order.)

## Out of scope / leave alone

- Commented historical `[0]/[1]` blocks
- Notebooks / docstrings
- `find_LongShortGlobal_epoch_names` `len==3` branch (`[-3]/[-2]/[-1]`) — only bites if exactly three completed results are odd/odd/any; directional pipelines have 9. Note only; do not change unless we hit it.

## Verification

On a kdiba directional session, after reload:

```python
curr_active_pipeline.safe_resolve_PrimarySecondaryGlobal_epoch_names()
# expect maze1_any, maze2_any, maze_any (not maze1_odd / maze2_odd)
```

Re-run BatchPhoJonathan / overlap display; aclu 55 short PF should resolve without `not present in pf1d_short` for the canonical short track.
