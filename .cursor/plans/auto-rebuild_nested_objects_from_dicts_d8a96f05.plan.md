---
name: Auto-rebuild nested objects from dicts
overview: Create a recursive function that automatically rebuilds nested objects from dicts loaded by `loadSplitData` using type annotations from attrs classes, eliminating the need for manual `_helper_rebuild_obj_from_class_if_needed` calls.
todos: []
---

# Automatic Object Rebuilding from Dicts Using Type Annotations

## Problem

`loadSplitData` returns a dictionary where all values (including nested objects) are dicts instead of their proper types. Currently, users must manually call `_helper_rebuild_obj_from_class_if_needed` for each nested object, which is error-prone and tedious.

## Solution

Enhance `_helper_rebuild_obj_from_class_if_needed` to automatically rebuild nested objects by:

1. Using `attrs.fields()` to get field definitions from attrs classes
2. Using `typing.get_type_hints()` to extract type annotations
3. Recursively processing nested structures (Optional, Dict, List, etc.)
4. Automatically rebuilding nested attrs objects when their values are dicts

## Implementation Details

### New Function: `_rebuild_nested_objects_from_dict`

Create a new helper function that:

- Takes a target class and a dict value
- Uses `attrs.fields()` to iterate through all fields
- Uses `typing.get_type_hints()` to get type annotations for each field
- For each field:
  - Extracts the actual type (handling Optional, Dict, List, etc.)
  - If the field value is a dict and the type is an attrs class, recursively rebuild it
  - Handles nested Dict types (e.g., `Dict[str, PredictiveDecoding]`)
  - Handles nested List types (e.g., `List[DecodingLocalityMeasures]`)
  - Handles Optional types by extracting the inner type

### Enhanced `_helper_rebuild_obj_from_class_if_needed`

Modify the existing function to:

- First rebuild the top-level object (existing behavior)
- Then call `_rebuild_nested_objects_from_dict` to rebuild nested objects
- Maintain backward compatibility with existing code

### Type Extraction Utilities

Create helper functions to:

- Extract the actual type from `Optional[T] `→ `T`
- Extract value types from `Dict[K, V] `→ `V`
- Extract element types from `List[T] `→ `T`
- Check if a type is an attrs class using `attrs.has()`
- Handle forward references and string annotations

## Files to Modify

1. **[pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py](pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py)**

   - Add `typing.get_type_hints` import
   - Add `attrs.fields` import (already available via `attrs`)
   - Create `_extract_actual_type()` helper to handle Optional, Dict, List, etc.
   - Create `_is_attrs_class()` helper to check if a type is an attrs class
   - Create `_rebuild_nested_objects_from_dict()` recursive function
   - Enhance `_helper_rebuild_obj_from_class_if_needed()` to use the new recursive rebuilding

## Key Considerations

1. **Type Annotation Extraction**: Handle `Optional[T]`, `Dict[K, V]`, `List[T]`, and other generic types
2. **Forward References**: Handle string annotations that may need evaluation
3. **Circular Dependencies**: Prevent infinite recursion with a depth limit or visited set
4. **Performance**: Only process fields that are actually dicts to avoid unnecessary work
5. **Backward Compatibility**: Existing code using `_helper_rebuild_obj_from_class_if_needed` should continue to work
6. **Error Handling**: Gracefully handle cases where type annotations are missing or invalid

## Example Usage

After implementation, users can simply do:

```python
container = loadSplitData(split_save_folder, debug_print=True)
if isinstance(container, dict):
    container = _helper_rebuild_obj_from_class_if_needed(PredictiveDecodingComputationsContainer, container)
    # Nested objects (predictive_decoding, locality_measures, etc.) are automatically rebuilt
```

Instead of the current manual approach:

```python
container = _helper_rebuild_obj_from_class_if_needed(PredictiveDecodingComputationsContainer, container)
if isinstance(container.predictive_decoding, dict):
    container.predictive_decoding = _helper_rebuild_obj_from_class_if_needed(PredictiveDecoding, container.predictive_decoding)
if isinstance(container.predictive_decoding.locality_measures, dict):
    container.predictive_decoding.locality_measures = _helper_rebuild_obj_from_class_if_needed(DecodingLocalityMeasures, container.predictive_decoding.locality_measures)
```