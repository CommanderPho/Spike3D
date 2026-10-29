---
name: Refactor notebook helpers into package
overview: Move the layer-geometry, document-backup and layer-replacement helpers from the notebook into `vectorcraft_scripting` (general primitives as `LiveVectorCraft` methods, pure file/geometry helpers in a new module, the two composite workflows in a new `layers.py`), collapsing single-use helpers into `_subfn_*`/inlined code, then reduce the notebook cells to data + calls.
todos:
  - id: live-methods
    content: Add document_path/wait_for_background/save_copy/backup_document/find_by_name/node_bounds/unique_name/duplicate/place_file/undo_group methods to LiveVectorCraft (nested _subfn_ for unique_backup_path and resolve_duplicate_command)
    status: completed
  - id: find-measure
    content: Add find.bounds_xywh (make bounds_size use it) and new measure.py with intrinsic_size_pts (+ nested _subfn_svg_length_to_pt)
    status: completed
  - id: layers-module
    content: Create layers.py with layer_geometry and replace_layers (lazy pandas, nested _subfn_uniform_fit_rect / flatten / geom_row)
    status: completed
  - id: exports-pyproject-readme
    content: Update __init__.py exports, add pandas 'tables' extra to pyproject.toml, document new live helpers in README
    status: completed
  - id: notebook
    content: Replace the three notebook helper cells with data + calls using the new API
    status: completed
  - id: verify
    content: Import smoke test and read-only checks against the live app (document_path, find_by_name, layer_geometry, intrinsic_size_pts)
    status: completed
isProject: false
---

# Refactor notebook helpers into `vectorcraft_scripting`

## Source of truth

Notebook [VectorCraft_Scripting.ipynb](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\notebooks\VectorCraft_Scripting.ipynb), cells "Get specific layers" (`get_layer_geometry`), the backup cell (`active_document_path`, `unique_backup_path`, `wait_for_background_jobs`, `save_document_copy`, `backup_active_document`) and the replacement cell (`_subfn_svg_length_to_pt`, `source_file_size_pts`, `uniform_fit_rect`, `find_layer_nodes`, `node_bounds`, `unique_backup_name`, `resolve_duplicate_command`, `duplicate_nodes`, `place_replacement`, `replace_figure_layers`). 15 top-level functions today; after the refactor the package exposes 3 module-level functions, the rest become `LiveVectorCraft` methods or nested `_subfn_*`.

## 1. `LiveVectorCraft` methods — [live.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\live.py)

General, live-document primitives (each takes `self`, drops the `live` parameter):

- `document_path() -> Path` (from `active_document_path`; `document.inspect {depth:0, childLimit:0}` → `path`, assert when unsaved).
- `wait_for_background(timeout_sec=120.0, poll_sec=0.25)` (from `wait_for_background_jobs`; polls `ui.inspect` → `background`).
- `save_copy(out_path, timeout_sec=120.0) -> Dict` (from `save_document_copy`; `file.saveCopy`, handles `'cancelled'` / `pending` / `background`, asserts file exists; `warnings` printed as today).
- `backup_document(date_str=None, backup_subfolder='_BAK', also_native_copy=True) -> Dict[str, Path]` (from `backup_active_document`). `unique_backup_path` is used twice → nested `_subfn_unique_backup_path`. `date_str=None` → today's `%Y-%m-%d` (replaces the notebook `TODAY_DAY_DATE` global).
- `find_by_name(name, exact=True, limit=100) -> List[Dict]` (from `find_layer_nodes`; `document.find {name}` with client-side exact filter).
- `node_bounds(node_id) -> Dict[str, float]` (from `node_bounds`; `document.node summary` → xywh via `bounds_xywh`, see §2).
- `unique_name(base_name) -> str` (from `unique_backup_name`; loops `find_by_name(..., exact=True)`).
- `duplicate(ids) -> List` (from `duplicate_nodes`). `resolve_duplicate_command` is single-use → nested `_subfn_resolve_duplicate_command`, result cached on `self._duplicate_command` so the catalogue is fetched once per connection (same candidates `layer.duplicate`/`object.duplicate`/`edit.duplicate`, fallback `object.move {dx:0, dy:0, copy:true}`; new ids from reply else from `selection`).
- `place_file(path, rect, name=None, crop='crop', link=False) -> Tuple[List, Dict[str, float]]` (from `place_replacement`; `file.place` with `rect` as `[x,y,w,h]`, optional `object.setProps {name}`, returns `(ids, actual_bounds)`).
- `undo_group(label)` context manager (`edit.undoGroup.begin {label}` / `.end`), replacing the notebook's manual try/finally.

Keep two blank lines between methods; signatures on one line.

## 2. Pure helpers — [find.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\find.py) and new `measure.py`

- `find.bounds_xywh(bounds) -> Optional[Dict[str, float]]` (from `_subfn_bounds_as_xywh`): accepts the dict form live `document.inspect` returns and the `[x,y,w,h]` list form. Rewrite existing `bounds_size` to delegate to it so `ObjectQuery(min_width=...)` filters start matching live dict bounds (today they silently never match — this is why the notebook's `Path matches: 0`).
- New `measure.py` with one public function `intrinsic_size_pts(path) -> Optional[Tuple[float, float]]` (from `source_file_size_pts`): SVG `width`/`height` (units table) → `viewBox` fallback; `.vectorcraft` artboard rect; else `None`. `_subfn_svg_length_to_pt` is used twice inside → stays nested; the `_SVG_LENGTH_TO_PT` table becomes a module constant.

## 3. Composite workflows — new `layers.py`

- `layer_geometry(live, layer_names) -> pd.DataFrame` (from `get_layer_geometry`): same find → `inspect(ids)` → tree-flatten fallback → `get_node` logic. Keep `_subfn_flatten_layer_nodes` and `_subfn_node_geom_row` nested; `_subfn_bounds_as_xywh` replaced by `find.bounds_xywh`. Drop the `if live is None: connect_first()` (caller passes a live).
- `replace_layers(live, layer_rects, source_folder, *, expect_title=None, backup_suffix=None, new_suffix='_PROGRAMMATIC', hide_original=True, hide_backup=True, allow_non_uniform_scaling=False, aspect_tolerance=0.01, link=False) -> pd.DataFrame` (from `replace_figure_layers`): identical per-layer flow (assert file exists → `find_by_name` → `unique_name` + `duplicate` + `setProps` backup → aspect check via `intrinsic_size_pts` → `place_file` → row with target/placed/actual/d* columns). `uniform_fit_rect` is single-use → nested `_subfn_uniform_fit_rect`. `expect_title=None` skips the title assert; `backup_suffix=None` → today's date.
- pandas is imported lazily inside these two functions (not a package dependency); add `tables = ["pandas"]` to `[project.optional-dependencies]` in [pyproject.toml](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\pyproject.toml).

## 4. Exports and docs

- [__init__.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\src\vectorcraft_scripting\__init__.py): export `bounds_xywh`, `intrinsic_size_pts`, `layer_geometry`, `replace_layers`.
- [README.md](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\EXTERNAL\VectorCraft_Scripting\README.md): extend the "Live helpers" table with the new methods and a short "Replace figure layers" example (backup → `undo_group` → `replace_layers`).

## 5. Notebook

Replace the three helper cells with data + calls (notebook is in context, so it is editable):

```python
from vectorcraft_scripting import layer_geometry, replace_layers

layer_geom_df = layer_geometry(live, layer_names)
curr_fig_geom_dict = layer_geom_df[['x', 'y', 'width', 'height']].to_dict(orient='index')
```

```python
if ('live' not in globals()) or (live is None) or (not live.is_connected):
    live = connect_first()
backup_paths = live.backup_document()
with live.undo_group(f'Replace figure layers ({datetime.now():%Y-%m-%d})'):
    replacement_df = replace_layers(live, curr_fig_geom_dict, curr_replacement_search_path, expect_title=active_figure_name)
replacement_df
```

`figs_layer_geometry` / `replacement_paths_dict` / `active_figure_name` stay in the notebook as data.

## 6. Verification

- Import smoke test from `/tmp` with `PYTHONPATH=.../src`: `python -c "import vectorcraft_scripting as v; print(v.__all__)"`.
- Against the running app (read-only calls): `live.document_path()`, `live.find_by_name('ScatterOverTime_Laps_perTimeBin.svg')`, `live.node_bounds(<id>)`, `layer_geometry(live, [...])`, `intrinsic_size_pts(<svg>)` on one of the 2026-10-29 SVGs (expect ~388.5×157 pt). No `replace_layers` / `backup_document` run without the user.