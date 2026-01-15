---
name: Fix recursive dict conversion in safeSaveSplitData
overview: Modify `safeSaveSplitData` to only convert objects to dictionaries when they fail to pickle, preserving original types for picklable objects and their nested structures.
todos: []
---

# Fix Recursive Dict Conversion in safeSaveSplitData

## Problem

The `safeSaveSplitData` function is recursively converting all attrs objects to dictionaries using `asdict()`, even when they are successfully picklable. This causes type information to be lost, complicating later loading operations.

## Solution

Implement a "try-pickle-first" strategy that only converts objects to dicts when pickling fails, preserving original types for picklable objects.

## Implementation Plan

### 1. Create Helper Function: `_try_pickle_or_convert_to_dict()`

- **Location**: Add before `safeSaveSplitData` function (around line 145)
- **Purpose**: Test if an object is picklable before converting to dict
- **Logic**:
- Try to pickle the object using `pickle.dumps()` in a try-except block
- If pickling succeeds, return the object as-is
- If pickling fails with `PicklingError`, convert to dict:
- For attrs objects: use `asdict()` (but this will recursively convert nested objects)
- For objects with `__dict__`: use `v.__dict__`
- For primitive types: return as-is
- **Note**: This function will handle the top-level conversion, but nested objects within the converted dict will still be converted by `asdict()`. We'll need a more sophisticated recursive approach.

### 2. Create Recursive Helper: `_convert_unpicklable_to_dict()`

- **Location**: Add before `safeSaveSplitData` function
- **Purpose**: Recursively convert only unpicklable objects to dicts, preserving picklable nested objects
- **Logic**:
- For each value in a structure (dict, list, tuple):
- Test if it's picklable
- If picklable: keep as-is
- If not picklable: convert to dict (attrs) or `__dict__` (regular object)
- Recursively process nested structures
- **Handles**: dicts, lists, tuples, attrs objects, regular objects

### 3. Modify `safeSaveSplitData()` Function

- **Location**: [Loading.py:149-279](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py)
- **Changes**:
- **Line 186-198**: Replace the top-level conversion logic:
- Instead of immediately calling `asdict(computed_data)`, first try to pickle it
- Only convert to dict if pickling fails
- Use the recursive helper to preserve nested picklable objects
- **Line 243-251**: Replace the value conversion logic:
- Instead of immediately converting attrs objects to dict, first try to pickle
- Only convert if pickling fails
- Use the recursive helper for nested structures

### 4. Implementation Details

The key insight is that we need to:

1. Test picklability at each level
2. Only convert when necessary
3. Recursively process nested structures while preserving picklable types

**Example flow**:

- Object A (attrs) contains Object B (attrs, picklable) and Object C (attrs, not picklable)
- Test A: if picklable → save as-is
- If A not picklable → convert A to dict, but:
- Test B: if picklable → keep B as original type in the dict
- Test C: if not picklable → convert C to dict

### 5. Edge Cases to Handle

- Circular references (use visited set)
- Maximum recursion depth
- Primitive types (int, str, float, bool, None)
- NumPy arrays and pandas DataFrames (usually picklable)
- Lists and tuples containing mixed picklable/unpicklable objects
- Nested dicts with picklable/unpicklable values

## Files to Modify

- `pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py`
- Add helper functions before `safeSaveSplitData` (around line 145)
- Modify `safeSaveSplitData` function (lines 149-279)

## Testing Considerations

- Test with objects that are fully picklable (should remain unchanged)
- Test with objects that have some picklable and some unpicklable nested objects
- Test with deeply nested structures
- Verify that loading with `loadSplitData` works correctly with preserved types