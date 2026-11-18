---
name: KDiba RG0 Compat
overview: Make minimal comment-out-then-replace edits in `ReviewOfWork_KDiba_Compat.ipynb` run-group-0 cells so they match `release/pho-diba-2025-paper` APIs. Audit shows one real incompatibility; other RG0 cells already match release.
todos:
  - id: cell3-path-widget
    content: "In ReviewOfWork_KDiba_Compat.ipynb cell 3: comment out required_relative_path=... call; add 2-arg release-compatible build_global_data_root_parent_path_selection_widget line"
    status: pending
  - id: verify-rg0-scan
    content: Quick re-scan run-group-0 cells for any remaining required_relative_path or other develop-only kwargs; confirm none remain live
    status: pending
isProject: false
---

# Adapt run-group-0 for release/pho-diba-2025-paper

## Context

- Target notebook: [`ReviewOfWork_KDiba_Compat.ipynb`](h:\TEMP\Spike3DEnv_KDibaVersion\Spike3D\ReviewOfWork_KDiba_Compat.ipynb) (copied from develop-era [`ReviewOfWork_2026-04-01.ipynb`](h:\TEMP\Spike3DEnv_KDibaVersion\Spike3D\ReviewOfWork_2026-04-01.ipynb)).
- Dependent libs are already on `release/pho-diba-2025-paper` (pyPhoPlaceCellAnalysis / pyPhoCoreHelpers / NeuroPy).
- `run-group-0` cells: **1, 3, 27, 30, 32, 33, 35**.
- Edit style (per request): **comment out the current line, then add the release-compatible replacement** on the next line(s). Do not silently delete the old call.

## Audit result (concrete)

Against checked-out release code, almost every RG0 symbol already exists (`PipelinePickleFileSelectorWidget`, `CustomProcessingPhases`, `BatchPlotting`, `Assert`, NonPBE/WCorr imports, load/evaluate/compute kwargs used in RG0, etc.).

**Only hard post-release API gap in RG0:**

[`build_global_data_root_parent_path_selection_widget`](h:\TEMP\Spike3DEnv_KDibaVersion\pyPhoCoreHelpers\src\pyphocorehelpers\gui\Jupyter\simple_widgets.py) on release is **2-arg only**:

```python
def build_global_data_root_parent_path_selection_widget(all_paths: List[Path], on_user_update_path_selection: Callable):
```

HEAD commit of the compat notebook still has the develop call in **cell 3**:

```python
global_data_root_parent_path_widget = build_global_data_root_parent_path_selection_widget(all_paths, on_user_update_path_selection, required_relative_path=session_relative_path)
```

Working tree already partially commented the kwarg; replace that awkward multi-line with the explicit comment-out + replace pattern below.

**Keep as-is (release-correct, do not revert to April paper form):**

- Cell 1 `BatchPlotting` imports (paper’s free `batch_*_programmatic_figures` imports would fail on current release tip).
- Cells 27/30/32/33/35 computation/load APIs (match release signatures).

## Edit (cell 3 only)

In the Load Data cell (`run-group-0`), replace the path-widget construction with:

```python
## Choose the first data root that contains this specific session folder (not merely an extant Data root):
session_relative_path = Path('KDIBA').joinpath(curr_context.animal, curr_context.exper_name, curr_context.session_name)
# global_data_root_parent_path_widget = build_global_data_root_parent_path_selection_widget(all_paths, on_user_update_path_selection, required_relative_path=session_relative_path)  # develop-only kwarg; not on release/pho-diba-2025-paper
global_data_root_parent_path_widget = build_global_data_root_parent_path_selection_widget(all_paths, on_user_update_path_selection)
display(global_data_root_parent_path_widget)
```

Keep the existing `session_relative_path` / `basedir.exists()` assert — that is notebook-side validation, not a library kwarg.

No other `run-group-0` cells need API edits for release compatibility.

## Out of scope

- Non–run-group-0 cells
- Pickle migration / library changes (separate from notebook RG0 API surface)
- Reverting `BatchPlotting` to paper free-function imports
- Changing `fail_on_exception` defaults
