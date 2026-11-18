---
name: Default debug kwarg False
overview: Change the default value of the debug-related keyword argument `enable_debug_outputs` from `True` to `False` in all function signatures where it is defined, and update the docstring that describes the default.
todos: []
isProject: false
---

# Change default `enable_debug_outputs` to False

## Scope

Update only **default parameter values** and the **docstring** that documents the default. Call sites that explicitly pass `enable_debug_outputs=True` (e.g. [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) line 3694, [PendingNotebookCode.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) line 1169) are left unchanged so they continue to get debug output when desired.

## Edits

### 1. [PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)

- **Line 668** (signature of `compute_locality_measures_for_posterior`): change `enable_debug_outputs: bool = True` to `enable_debug_outputs: bool = False`.
- **Line 680** (docstring): change `(default: True)` to `(default: False)` for the `enable_debug_outputs` parameter.
- **Line 892** (signature of `compute_locality_measures`): change `enable_debug_outputs: bool=True` to `enable_debug_outputs: bool=False`.

### 2. [SequenceBasedComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\SequenceBasedComputations.py)

- **Line 1466** (signature that includes `enable_debug_outputs`): change `enable_debug_outputs:bool=True` to `enable_debug_outputs:bool=False`.

## Result

- Default behavior: no debug dicts are computed or stored, reducing memory use (especially for the second-session kernel crash case).
- Explicit `enable_debug_outputs=True` at call sites still enables debug where needed.

