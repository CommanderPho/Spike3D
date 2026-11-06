---
name: deepcopy surprise epoch labels
overview: Stop fig_surprise_results from mutating shared sess.epochs by deepcopying before renaming labels to long/short, so figure_run cannot corrupt find_LongShortGlobal_epoch_names even when pf_dt_sequential_surprise is present in old pickles.
todos:
  - id: worktree
    content: Create plan/deepcopy-surprise-epoch-labels branch + sibling worktree for pyPhoPlaceCellAnalysis
    status: completed
  - id: deepcopy-fix
    content: Add deepcopy import and deepcopy sess.epochs before label rename in fig_surprise_results
    status: completed
  - id: smoke-verify
    content: Load pickle with surprise present; call fig_surprise_results; confirm paradigm labels and find_LongShortGlobal_epoch_names unchanged
    status: cancelled
isProject: false
---

# Guard fig_surprise_results epoch-label mutation

## Why

[`fourthYearPresentation.py` L338-341](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/fourthYearPresentation.py) assigns `['long','short']` onto `curr_active_pipeline.sess.epochs._df['label']` in place. That object is the live session paradigm; later completion functions then look for `long_any`/`short_any` and assert.

The phase3 includelist change stops **new** pipelines from computing surprise. This deepcopy protects **existing** pickles that still contain `extended_stats['pf_dt_sequential_surprise']`, and keeps the plot working without side effects if surprise is re-enabled later.

## Change (single file)

In [`fourthYearPresentation.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/fourthYearPresentation.py):

1. Add `from copy import deepcopy` near the existing imports.
2. Replace the live reference with a deep copy before the label rename:

```python
# Rename track labels only on a plot-local copy — never mutate sess.epochs/paradigm (breaks find_LongShortGlobal_epoch_names for later completion fns)
track_epochs = deepcopy(curr_active_pipeline.sess.epochs)
assert len(track_epochs._df['label']) == 2, f"labels should be ['maze1', 'maze2']!"
track_epochs._df['label'] = ['long', 'short']
```

`plot_data_and_epochs(..., track_epochs, ...)` continues to receive the renamed copy for region drawing; `sess.epochs` stays `maze1`/`maze2`.

## Git

Per workspace rule: sibling worktree of `pyPhoPlaceCellAnalysis` on `plan/deepcopy-surprise-epoch-labels` (or extend the existing `plan/batch-phase3-surprise-removal` worktree if you prefer one branch — default: **new branch + worktree** so this guard is a separate, reviewable commit). Do not switch the IDE-open `develop` checkout.

## Verify

Load a pickle that still has `pf_dt_sequential_surprise`, call `fig_surprise_results(curr_active_pipeline)`, then assert:

- `list(curr_active_pipeline.sess.paradigm.get_unique_labels()) == ['maze1', 'maze2']`
- `curr_active_pipeline.find_LongShortGlobal_epoch_names() == ('maze1_any', 'maze2_any', 'maze_any')`
- function returns graphics (does not KeyError)

No full clean→figure chain required for this guard.