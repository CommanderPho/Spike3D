---
name: Implement loadSplitData function
overview: Implement the `loadSplitData` function as the reciprocal of `safeSaveSplitData`. It should load split pickle files from a folder and return a dictionary of the loaded data.
todos:
  - id: "1"
    content: Implement path resolution logic to determine split folder location (mirroring safeSaveSplitData logic)
    status: completed
  - id: "2"
    content: Implement file discovery using rglob to find all Split_*.pkl files
    status: completed
  - id: "3"
    content: Implement loading logic with error handling for each split file
    status: completed
  - id: "4"
    content: Handle tuple format (v_dict, module, name) that was saved by safeSaveSplitData
    status: completed
  - id: "5"
    content: Add function attributes decorator and complete docstring
    status: completed
---

# Implement loadSplitData function

## Overview

Create a standalone `loadSplitData` function that is the reciprocal of `safeSaveSplitData`. It should load split pickle files from a folder and reconstruct the data dictionary.

## Function Signature

```python
def loadSplitData(pkl_path: Union[str, Path], debug_print:bool=True, **kwargs) -> Dict[str, Any]:
```

## Implementation Details

### Path Resolution Logic (mirroring safeSaveSplitData)

1. Resolve `pkl_path` to a Path object
2. If `pkl_path` is a directory:

   - Use default filename "computed_data.pkl" 
   - Append "_split" to get the split folder name

3. If `pkl_path` is a file:

   - Use the file's stem (name without extension)
   - Append "_split" to get the split folder name
   - The split folder is in the same directory as the file

### Loading Logic (based on try_load_split_pickled_global_computation_results)

1. Find all `Split_*.pkl` files in the split folder using `rglob`
2. For each file:

   - Extract the key name by removing "Split_" prefix from the stem
   - Load the file using `loadData(p, **kwargs)`
   - Handle the loaded value:
     - If it's a tuple with 3 elements: `(v_dict, module_name, type_name)` - extract the dict
     - If it's already a dict: use it directly
     - Store in the result dictionary with the extracted key

3. Handle errors gracefully (continue loading other files if one fails)

### Return Value

Return a dictionary mapping keys to loaded values (or their dict representations if they were saved as tuples with type info).

### Files to Modify

- [`pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py`](pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py) - Implement `loadSplitData` function at line 561