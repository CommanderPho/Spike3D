---
name: Vispy scene tree widget
overview: Add a `VispySceneTreeWidget` Qt widget (QTreeWidget-based) to `VispyHelpers` that displays the vispy scene graph hierarchy for any `SceneCanvas`, with per-node metadata (type, name, visible, order, opacity, transform) and interactive controls (toggle visibility, refresh).
todos:
  - id: tree-widget-class
    content: Implement VispySceneTreeWidget(QWidget) with QTreeWidget, recursive _populate(), visibility toggle, refresh button
    status: completed
  - id: classmethod-factory
    content: Add VispyHelpers.create_scene_tree_widget() classmethod factory
    status: completed
  - id: example-function
    content: Add example_scene_tree_widget() standalone example at bottom of file
    status: completed
isProject: false
---

# Vispy Scene Tree Widget

## Context

There is no scene graph inspector in the codebase. The vispy `Node` class exposes `.children`, `.name`, `.visible`, `.order`, `.opacity`, `.transform`, and `describe_tree()`. The existing `[DockAreaDocksTree](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\DockAreaDocksTree\DockAreaDocksTree.py)` provides a precedent QTreeWidget pattern using `QTreeWidgetItem`.

## Target file

`[vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)` -- add the widget class and a `@classmethod` factory on `VispyHelpers`.

## Implementation

### 1. `VispySceneTreeWidget(QtWidgets.QWidget)` class (~line 440, before `VispyHelpers`)

A standalone Qt widget wrapping a `QTreeWidget` that recursively walks a vispy `Node` and displays the scene hierarchy.

**Columns:** Type | Name | Visible | Order | Opacity | Transform

**Key details:**

- Constructor takes `root_node: Node` (typically `canvas.scene`) and optional `canvas: scene.SceneCanvas`
- `_populate(node, parent_item)` -- recursive walk using `node.children`; creates `QTreeWidgetItem` per node
- Each item stores a reference to the vispy `Node` via `item.setData(0, Qt.UserRole, node)`
- Visible column uses checkboxes; toggling calls `node.visible = checked` then `canvas.update()` if canvas is available
- A "Refresh" button re-walks the tree (`clear()` + `_populate()`)
- The tree starts expanded to depth 3

**Node display string:** `type(node).__name__` (e.g. `Line`, `ViewBox`, `SubScene`, `Grid`), with `node.name` shown in the Name column if set.

### 2. `VispyHelpers.create_scene_tree_widget()` classmethod

```python
@classmethod
def create_scene_tree_widget(cls, canvas: scene.SceneCanvas, parent: Optional[QtWidgets.QWidget] = None) -> 'VispySceneTreeWidget':
```

Convenience factory that creates a `VispySceneTreeWidget` rooted at `canvas.scene`, with the canvas reference for live updates.

### 3. Standalone example function

`example_scene_tree_widget()` at the bottom of the file, following the pattern of `example_viewport_overlay_text()` -- creates a canvas with some visuals, then shows the tree widget in a separate window.

## Design decisions

- Pure Qt (via `qtpy` which is already imported) -- no `.ui` file needed, built programmatically like the simpler widgets in the codebase
- Widget is decoupled from any specific vispy canvas class so it works with any `SceneCanvas`
- Checkbox-driven visibility toggle provides immediate interactive value
- `UserRole` data storage for node references follows the `DockAreaDocksTree` pattern

