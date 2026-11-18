---
name: GL blend mode combo column
overview: Replace the current "Mode" column (showing draw primitive mode) with a "GL Blend" column that displays the OpenGL blending preset (opaque/translucent/additive) and uses a QComboBox delegate for editing on supported visual nodes.
todos:
  - id: rename-column
    content: Rename 'Mode' to 'GL Blend' in _column_headers
    status: completed
  - id: render-gl-blend
    content: Replace _render_mode with _render_gl_blend reading node._vshare.gl_state preset
    status: completed
  - id: has-gl-blend
    content: Replace _node_has_editable_mode to check hasattr(node, 'set_gl_state')
    status: completed
  - id: combo-delegate
    content: Add _BlendPresetDelegate(QStyledItemDelegate) with QComboBox for opaque/translucent/additive
    status: completed
  - id: install-delegate
    content: Install delegate on column 5 in _init_ui
    status: completed
  - id: on-item-changed
    content: Update _on_item_changed column 5 to call node.set_gl_state preserving extra kwargs
    status: completed
  - id: cleanup
    content: Remove _VALID_MESH_MODES, _VALID_LINE_METHODS
    status: completed
isProject: false
---

# Replace Mode column with GL Blend combo-box column

File: [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)

## Key vispy internals

- Visuals store blending state in `node._vshare.gl_state`, which is a dict like `{'preset': 'translucent', 'depth_test': True, ...}`
- The preset name is at `node._vshare.gl_state.get('preset', None)` and can be `'opaque'`, `'translucent'`, or `'additive'` (or `None` if never set)
- Only nodes with `set_gl_state` method (i.e. Visual subclasses) support this; pure `Node` objects do not
- To change the preset: call `node.set_gl_state('translucent')` -- this replaces the entire gl_state dict. To preserve extra kwargs (like `depth_test`, `cull_face`), we should read the current dict, update just the preset, and call `set_gl_state(**updated_dict)`

## Changes

### 1. Rename column from "Mode" to "GL Blend"

- In `__init`__ line 544: change `'Mode'` to `'GL Blend'` in `_column_headers`

### 2. Replace `_render_mode` with `_render_gl_blend`

- In `_get_default_column_renderers` (line 608-613): replace the `_render_mode` function with `_render_gl_blend` that reads `node._vshare.gl_state.get('preset', '')` for nodes with `set_gl_state`, and returns `''` otherwise
- Update the returned dict key from `'Mode'` to `'GL Blend'`

### 3. Replace `_node_has_editable_mode` with `_node_has_gl_blend`

- Line 646-648: check `hasattr(node, 'set_gl_state')` instead of `hasattr(node, 'mode') or hasattr(node, 'method')`

### 4. Update `_populate` to use 'GL Blend' column name

- Line 657: change `'Mode'` to `'GL Blend'` in `_get_cell_text` call

### 5. Add a `QStyledItemDelegate` subclass for combo-box editing

Define a small `_BlendPresetDelegate(QStyledItemDelegate)` class (inside the file, before `VispySceneTreeWidget` or as a nested class) that:

- `createEditor`: returns a `QComboBox` with items `['', 'opaque', 'translucent', 'additive']`
- `setEditorData`: sets the combo's current text from the item
- `setModelData`: writes the combo's current text back to the model

### 6. Install the delegate in `_init_ui`

- After creating `self.tree`, call `self.tree.setItemDelegateForColumn(5, _BlendPresetDelegate(self.tree))`

### 7. Update `_on_item_changed` for column 5

- Replace the current Mesh/Line mode handling (lines 693-702) with GL blend logic:
  - Read `new_text = item.text(5).strip()`
  - If `new_text` is one of `('opaque', 'translucent', 'additive')` and the node has `set_gl_state`:
    - Read existing extra kwargs from `node._vshare.gl_state` (excluding `'preset'`)
    - Call `node.set_gl_state(new_text, **extra_kwargs)` to preserve settings like `depth_test`
  - Call `canvas.update()`

### 8. Remove stale class attributes

- Remove `_VALID_MESH_MODES` and `_VALID_LINE_METHODS` (lines 674-675) -- no longer needed

