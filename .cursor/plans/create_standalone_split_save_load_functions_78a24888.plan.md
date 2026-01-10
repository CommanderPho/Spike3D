---
name: Create standalone split save/load functions
overview: Create standalone functions `safeSaveSplitData` and `loadSplitData` in Loading.py based on the class methods in Computation.py, but without class dependencies - accepting data and paths as parameters instead. Functions should match the signature style of standard `safeSaveData`/`loadData` and have no references to "global_computation_results".
todos:
  - id: "1"
    content: Replace the unfinished safeSaveSplitData implementation (lines 149-249) with a complete standalone version based on save_split_custom_results
    status: completed
  - id: "2"
    content: Add new loadSplitData function after safeSaveSplitData, based on try_load_split_pickled_global_computation_results
    status: completed
  - id: "3"
    content: Ensure all imports are present (PicklingError, print helpers, saveData/loadData)
    status: completed
  - id: "4"
    content: Verify function signatures match the plan and return types are correct, with no "global_computation_results" references
    status: completed
---

# Create Standalone Split Save/Load Functions

## Analysis

The class methods in `Computation.py` (lines 2928-3140) provide split save/load functionality:

- `save_split_custom_results` - saves `global_computation_results.computed_data` to split pickle files
- `try_load_split_pickled_global_computation_results` - loads split pickle files from disk

These methods depend on class instance attributes (`self.global_computation_results.computed_data`, `self.global_computation_results_pickle_path`, `self.get_output_path()`).

## Goal

Create standalone functions that:

1. Accept data and paths as parameters (not class instance attributes)
2. Provide the same functionality as the class methods
3. Can be used independently without a pipeline instance
4. Match the signature style of standard `safeSaveData`/`loadData` functions
5. Have NO references to "global_computation_results" or similar naming

## Implementation Plan

### Function 1: `safeSaveSplitData`

**Location**: [Loading.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Loading.py) - Replace the unfinished implementation at lines 149-249

**Signature**:

```python
def safeSaveSplitData(data: Dict[str, Any], pickle_path: Optional[Union[str, Path]] = None, output_path: Optional[Union[str, Path]] = None, override_pickle_path: Optional[Union[str, Path]] = None, override_pickle_filename: Optional[str] = None, include_includelist: Optional[List[str]] = None, continue_after_pickling_errors: bool = True, debug_print: bool = True) -> Tuple[Path, Dict[str, Path], Dict[str, type], List[str]]
```

**Key Changes from Class Method**:

- Replace `self.global_computation_results.computed_data` with parameter `data: Dict[str, Any]`
- Replace `self.global_computation_results_pickle_path` with parameter `pickle_path: Optional[Union[str, Path]]`
- Replace `self.get_output_path()` with parameter `output_path: Optional[Union[str, Path]]`
- Remove all references to "global_computation_results" or similar naming
- Use generic parameter names matching the style of `safeSaveData`/`loadData`
- Accept `Union[str, Path]` like standard functions
- Keep all other logic identical (path resolution, split folder creation, saving logic)

**Logic Flow**:

1. Resolve `pickle_path` using override logic:

   - If `override_pickle_path` provided: use it (file or directory)
   - Else if `override_pickle_filename` provided: use `output_path.joinpath(override_pickle_filename)`
   - Else: use `pickle_path` (must be provided)

2. Create split save folder: `{stem}_split` in parent directory
3. Iterate through `data.items()`
4. For each item in `include_includelist` (or all if None):

   - Save as `Split_{key}.pkl` with `(v_dict, module_str, type_name)` tuple using `saveData`
   - Handle `KeyError` and `PicklingError` exceptions

5. Return `(split_save_folder, split_save_paths, split_save_output_types, failed_keys)`

### Function 2: `loadSplitData`

**Location**: [Loading.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Loading.py) - Add after `safeSaveSplitData`

**Signature**:

```python
def loadSplitData(pickle_path: Union[str, Path], debug_print: bool = False, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Path], Dict[str, Path], List[Path]]
```

**Key Changes from Class Method**:

- Remove `@classmethod` decorator (standalone function)
- Use `pickle_path: Union[str, Path] `to match `loadData` signature style
- Add `**kwargs` to match `loadData` signature (pass through to `loadData`)
- Use `debug_print: bool = False` to match `loadData` default
- Remove all references to "global_computation_results" or similar naming
- Keep all loading logic identical

**Logic Flow**:

1. Resolve path (convert to Path if string, handle file vs directory)
2. Determine split folder: if path is file, use `{stem}_split`; if directory, use directly
3. Assert folder exists and is directory
4. Find all `Split_*.pkl` files using `rglob`
5. For each file:

   - Extract key from filename: `p.stem.removeprefix('Split_')`
   - Load using `loadData(p, debug_print=debug_print, **kwargs)` (pass through kwargs)
   - Track successes and failures

6. Return `(loaded_data, successfully_loaded_keys, failed_loaded_keys, found_split_paths)`

## Files to Modify

- [Loading.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Loading.py)
  - Replace lines 149-249 (the unfinished `safeSaveSplitData` implementation)
  - Add new `loadSplitData` function after `safeSaveSplitData`

## Implementation Details

### Dependencies Needed

- `from pickle import PicklingError` (already imported via other imports)
- `from pyphocorehelpers.print_helpers import print_filesystem_file_size, print_object_memory_usage` (already imported)
- `saveData` and `loadData` (already defined in same file)

### Path Resolution Logic (for `safeSaveSplitData`)

The function uses this priority:

1. If `override_pickle_path` provided:

   - If file: use directly
   - If directory: join with `override_pickle_filename` or default filename ("data.pkl")

2. Else if `override_pickle_filename` provided:

   - Use `output_path.joinpath(override_pickle_filename)` (output_path must be provided)

3. Else:

   - Use `pickle_path` (must be provided)

### Error Handling

- `KeyError`: print warning, skip item, continue
- `PicklingError`: if `continue_after_pickling_errors=True`, print warning and skip; else raise
- Track all failures in `failed_keys` list

### Return Values

- `safeSaveSplitData`: `(split_save_folder: Path, split_save_paths: Dict[str, Path], split_save_output_types: Dict[str, type], failed_keys: List[str])`
- `loadSplitData`: `(loaded_data: Dict[str, Any], successfully_loaded_keys: Dict[str, Path], failed_loaded_keys: Dict[str, Path], found_split_paths: List[Path])`

### Naming Conventions

- All function parameters use generic names (no "global_computation_results" references)
- Variable names in function bodies also use generic names:
  - `loaded_data` instead of `loaded_global_computation_results`
  - `computed_data` → `data`
  - `global_computation_results_pickle_path` → `pickle_path`
  - `override_global_pickle_path` → `override_pickle_path`
  - `override_global_pickle_filename` → `override_pickle_filename`
  - `global_computed_data` → `data` (already the parameter name)