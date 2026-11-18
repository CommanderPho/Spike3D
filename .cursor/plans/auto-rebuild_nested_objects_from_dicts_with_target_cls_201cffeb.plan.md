---
name: Auto-rebuild nested objects from dicts with target_cls
overview: ""
todos: []
---

# Automatic Object Rebuilding from Dicts Using Type Annotations

## Problem

`loadSplitData` returns a dictionary where all values (including nested objects) are dicts instead of their proper types. Currently, users must manually call `_helper_rebuild_obj_from_class_if_needed` for each nested object, which is error-prone and tedious.

## Solution

1. Enhance `_helper_rebuild_obj_from_class_if_needed` to automatically rebuild nested objects by using type annotations
2. Add an optional `target_cls` parameter to `loadSplitData` that, if provided, automatically rebuilds the top-level object and all nested objects

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

### Enhanced `loadSplitData`

Add an optional `target_cls=None` parameter that:

- If `target_cls` is provided and the loaded data is a dict:
- Calls `_helper_rebuild_obj_from_class