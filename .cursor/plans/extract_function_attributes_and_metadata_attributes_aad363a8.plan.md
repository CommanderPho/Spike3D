---
name: Extract function_attributes and metadata_attributes
overview: Enhance the code indexing notebook to extract both `@function_attributes` and `@metadata_attributes` decorator metadata for functions, methods, and classes, and add them as columns to the DataFrame.
todos:
  - id: add_metadata_extraction
    content: Add extract_metadata_attributes_metadata() function to extract @metadata_attributes decorator metadata
    status: completed
  - id: update_enrich_function
    content: Update enrich_with_runtime_metadata() to handle functions, methods, and classes, extracting both function_attributes and metadata_attributes
    status: completed
    dependencies:
      - add_metadata_extraction
  - id: handle_methods
    content: Add logic to properly extract methods from their parent classes for metadata extraction
    status: completed
    dependencies:
      - update_enrich_function
  - id: handle_classes
    content: Add logic to extract metadata from class objects (not just callable objects)
    status: completed
    dependencies:
      - update_enrich_function
---

# Extract function_attributes and metadata_attributes for Code Index

## Overview

The current code in `code_index_2025.ipynb` only extracts `@function_attributes` metadata for callable objects. We need to:

1. Add extraction of `@metadata_attributes` decorator
2. Extract both decorators for classes (not just callable objects)
3. Properly handle methods by accessing them through their parent class

## Implementation Details

### 1. Add metadata_attributes extraction function

Create `extract_metadata_attributes_metadata()` function similar to `extract_function_attributes_metadata()`, using:

- `is_decorated_with_metadata_attributes()` from `pyphocorehelpers.programming_helpers`
- `get_decorated_metadata_attributes()` from `pyphocorehelpers.programming_helpers`

### 2. Enhance enrich_with_runtime_metadata function

Update `enrich_with_runtime_metadata()` in [code_index_2025.ipynb](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\scripts\code_index_2025.ipynb) to:

- **For functions**: Extract both `function_attributes` and `metadata_attributes` (if present)
- **For methods**: 
- Parse the class name from `row['name'] `(format: `ClassName.method_name`)
- Get the class object from the module
- Get the method from the class
- Extract both decorators from the method
- **For classes**: 
- Get the class object from the module
- Extract both decorators from the class (not just when callable)

### 3. Column naming convention

- `function_attributes` → prefix with `func_attr_` (e.g., `func_attr_short_name`, `func_attr_tags`)
- `metadata_attributes` → prefix with `meta_attr_` (e.g., `meta_attr_short_name`, `meta_attr_tags`)

This distinguishes between the two decorator types in the DataFrame.

### 4. Handle edge cases

- Methods that don't exist (e.g., `__init__` might not be in the class dict)
- Classes that can't be imported
- Objects that have both decorators (should extract both)
- Objects that have neither decorator (should still add row with None values)

## Files to Modify

- [code_index_2025.ipynb](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\scripts\code_index_2025.ipynb): Add `extract_metadata_attributes_metadata()` and update `enrich_with_runtime_metadata()`