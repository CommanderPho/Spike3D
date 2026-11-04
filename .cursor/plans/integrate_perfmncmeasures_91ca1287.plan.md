---
name: Integrate PerfmncMeasures
overview: Port `PerfmncMeasures` release improvements from `release/pho-diba-2025-paper` onto `develop` in pyPhoPlaceCellAnalysis via a surgical edit to one class block, plus small compatibility tweaks you chose (deprecated `worst_only`, falsy `x_col_name`).
todos:
  - id: extract-release-class
    content: Extract PerfmncMeasures block from release/pho-diba-2025-paper AcrossSessionResults.py
    status: completed
  - id: replace-develop-class
    content: Replace develop PerfmncMeasures block; fix metadata tags and return type on find_and_load
    status: completed
  - id: compat-plot-per-session
    content: Add deprecated worst_only + falsy x_col_name handling in plot_per_session
    status: completed
  - id: smoke-import
    content: Smoke-import PerfmncMeasures via uv run python from pyPhoPlaceCellAnalysis
    status: completed
isProject: false
---

# Integrate release `PerfmncMeasures` into develop

## Context

- **Repo / branch:** [`pyPhoPlaceCellAnalysis`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis) on **`develop`** (current).
- **Source of truth:** `class PerfmncMeasures` on **`release/pho-diba-2025-paper`** in [`AcrossSessionResults.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py) (~lines 4257–4519 on release; ~4476–4656 on develop).
- **Do not** merge or cherry-pick the whole file: `develop` has many unrelated hunks since merge-base `26f673d14` (inst-FR accessor, archiver, visualizations, etc.). Release-only edits elsewhere (BETAMAX path, `copy_batch_output_figures`, `AcrossSessionsVisualizations` refactor) are **out of scope**.

```mermaid
flowchart LR
  releaseBranch[release/pho-diba-2025-paper]
  developBranch[develop]
  classBlock[PerfmncMeasures class block]
  releaseBranch -->|extract classBlock| classBlock
  classBlock -->|surgical replace + compat| developBranch
```

## What to port (release behavior)

| Area | Change |
|------|--------|
| Metadata | Fix tag typo `perfmnc_sesssion` → `perfmnc_session` on class and `find_and_load` decorator |
| `find_and_load_most_recent_parsed_csv_files_df` | Add `should_delete_broken_CSVs: bool = False`; skip empty CSVs; optional `unlink`; docstring linking to `export_csvs`; delegate fixup to `post_load_fixup_df` |
| **New** `post_load_fixup_df` | Move concat-after logic here; safe `sort_values` (only existing columns); add `animal` via `UserAnnotationsManager`; same epoch/session splits and `percent_correct_diff` |
| `plot_per_indiv` | `correct_score_col_name`; `px.scatter` + y-axis `[0,1]` on all facets; facet row titles with `animal` when present |
| `plot_per_session` | Replace `worst_only` branching with `x_col_name` + `correct_score_col_name` (bar vs scatter by score column name) |

**Dependency:** `post_load_fixup_df` imports `UserAnnotationsManager` from `neuropy.core.user_annotations` (already used elsewhere in this module on develop).

## Compatibility additions (your choices)

1. **`plot_per_session` — deprecated `worst_only`**
   - Signature: keep release params **`x_col_name=None`**, **`correct_score_col_name='worse_percent_correct'`**, and add **`worst_only: Optional[bool] = None`** (or `= True` with deprecation only when explicitly passed—prefer **`None` default** so release defaults win).
   - If `worst_only is not None`: emit `warnings.warn(..., DeprecationWarning)` and map:
     - `worst_only=True` → equivalent to default release path: scatter with `correct_score_col_name='worse_percent_correct'`, default x = `time_bin_size` (after falsy-x handling).
     - `worst_only=False` → bar path with `correct_score_col_name='percent_correct'`, default x = `parent_epoch_id`.
   - If caller passes both `worst_only` and `correct_score_col_name` / explicit plot kwargs, prefer explicit `correct_score_col_name` / `x_col_name` and warn that `worst_only` is ignored.

2. **Falsy `x_col_name`**
   - At start of plot branch logic: `if not x_col_name:` treat as `None` (so `ReviewOfWork_2026-04-01.ipynb` `x_col_name=''` gets default columns). Apply in both bar and scatter branches.

3. **Return type (small cleanup)**  
   - Fix `find_and_load_most_recent_parsed_csv_files_df` annotation from incorrect `-> pd.DataFrame` to `-> Tuple[pd.DataFrame, pd.DataFrame]` if `Tuple` is already imported in the file (minimal one-line fix while touching the method).

## Implementation steps

1. **Extract release class body** (read-only prep):
   ```powershell
   cd h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis
   git show release/pho-diba-2025-paper:src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py
   ```
   Copy from `@metadata_attributes ... class PerfmncMeasures` through the end of `plot_per_session` (just before `class AcrossSessionHelpers`).

2. **Replace on develop** in [`AcrossSessionResults.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py): swap the existing `PerfmncMeasures` block (starts ~4476) with the release version, then apply the two compat edits in `plot_per_session` only.

3. **Preserve develop surroundings:** Do not alter `ExportValueNameCleaner` (develop already has variant parsing docs/changes above the class) or `AcrossSessionHelpers` below.

4. **Verify**
   - `uv run python -c "from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import PerfmncMeasures; assert hasattr(PerfmncMeasures, 'post_load_fixup_df')"` from repo root.
   - Optional: run existing pytest subset if any import `AcrossSessionResults` (no dedicated `PerfmncMeasures` tests found).

5. **Notebooks:** No notebook edits in this task. After install/sync, [`ReviewOfWork_2026-04-01.ipynb`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ReviewOfWork_2026-04-01.ipynb) should work against the updated package (`post_load_fixup_df`, `correct_score_col_name`, empty `x_col_name`). Archive notebook calling `plot_per_session(..., time_bin_size=None)` remains valid via deprecated `worst_only` default mapping or implicit release scatter defaults.

## Out of scope

- Merging `release/pho-diba-2025-paper` into `develop`.
- Other release hunks in `AcrossSessionResults.py` (paths, `AcrossSessionsVisualizations`, imports).
- Changes in [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) from release commits `1d0de5c45` / `e4ea520a4` unless you ask separately.

## Risk

- **Low:** isolated class; callers in repo already mostly expect release API in review notebooks; only in-repo `worst_only` usage is the old develop definition itself.
