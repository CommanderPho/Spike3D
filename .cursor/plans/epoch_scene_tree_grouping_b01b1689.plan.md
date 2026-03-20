---
name: Epoch scene tree grouping
overview: Reduce Scene Tree clutter by parenting all epoch-scoped visuals (posterior contours/fills, extrusion walls/tubes, emphasis planes) under one `vispy.scene.Node` per epoch. The existing `VispySceneTreeWidget` already reflects the scene graph hierarchy, so no tree-widget changes are required for nesting.
todos:
  - id: epoch-node-helper
    content: Add epoch_group_nodes dict + _get_or_create_epoch_group_node() using scene.Node under self.view.scene
    status: completed
  - id: thread-parent-contours
    content: Add scene_parent to _build_posterior_contours_3d + add_posterior_contours; store scene_parent on contour dict
    status: completed
  - id: thread-parent-extrusion-plane
    content: Pass scene_parent into wall/tube builders and build_contour_extrusions; add scene_parent to add_emphasis_plane
    status: completed
  - id: wire-add-epoch-visuals
    content: In add_epoch_visuals, create epoch node and pass to contours + emphasis plane + (implicit) extrusions
    status: completed
isProject: false
---

# Epoch-grouped scene nodes for Volumetric 2D viewer

## Context

- `[VispySceneTreeWidget._populate](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py)` walks **actual** `node.children`, so any visual parented under a `scene.Node` will appear nested in the Qt tree.
- Today, epoch-related visuals in `[Volumentric2DTimeSeriesPlotter](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)` all use `parent=self.view.scene`, which produces a **flat** list (`Contour[t=k]`, `ContourFill[t=k]`, `IntTube[...]`, emphasis plane, etc.).
- VisPy **does not** cascade `visible` from a plain parent `Node` to child visuals (each `VisualNode.draw` only checks its own `visible`). So the main win is **collapsible grouping / discovery**. Bulk show/hide for the active epoch remains the responsibility of existing `[set_active_epoch](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)` + per-visual visibility; optionally a follow-up could add guarded propagation (not in this pass) to avoid fighting the epoch slider.

## Implementation (all changes in `predicitive_decoding_vispy.py`)

1. **Per-epoch group node**
  - Add a field such as `epoch_group_nodes: Dict[int, Any] = field(default=Factory(dict))` (or fold into `epoch_visual_groups` if you prefer a single structure).
  - Add `_get_or_create_epoch_group_node(self, epoch_idx: int) -> scene.Node` that:
    - Returns existing node if present.
    - Otherwise creates `scene.Node(parent=self.view.scene, name=f'Epoch {epoch_idx}')`, stores it, returns it.
  - Use plain `[scene.Node](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/.venv/lib/site-packages/vispy/scene/node.py)` (not `SubScene`) to avoid extra sub-camera/view behavior.
2. **Thread `scene_parent` through builders (default = current behavior)**
  - `**_build_posterior_contours_3d`**: add parameter `scene_parent` (default `self.view.scene`). Replace every `parent=self.view.scene` when creating `vz.Line` / `vz.Mesh` with that parent.
  - `**add_posterior_contours`**: add optional `scene_parent=None`; resolve `parent = scene_parent if scene_parent is not None else self.view.scene`, pass into `_build_posterior_contours_3d`, and store `'scene_parent': parent` in `posterior_contours_by_key[identifier]` so downstream code knows the group root.
3. **Extrusions follow the same parent**
  - `**_build_extrusion_wall_mesh`** and `**_build_intersection_tube`**: add required or defaulted `scene_same_parent` argument; use it instead of hard-coded `self.view.scene`.
  - `**build_contour_extrusions`**: read `scene_parent = contour_entry.get('scene_parent', self.view.scene)` and pass it into wall/tube construction so `ExtWall[...]` and `IntTube[...]` sit under the same epoch node as the contour lines/fills.
4. **Emphasis plane under the same epoch**
  - `**add_emphasis_plane`**: add optional `scene_parent=None`; use resolved parent for `plane_mesh` and `edge_line` instead of `self.view.scene`.
5. **Wire `add_epoch_visuals`**
  - At the start, `epoch_node = self._get_or_create_epoch_group_node(epoch_idx)`.
  - Call `add_posterior_contours(..., scene_parent=epoch_node, ...)`.
  - Call `add_emphasis_plane(..., scene_parent=epoch_node, ...)`.
  - `build_contour_extrusions` unchanged at the call site; it picks up the stored parent from the contour entry.
6. **Backwards compatibility**
  - Notebook/docstring-style calls that invoke `add_posterior_contours` without `scene_parent` keep attaching to `self.view.scene` (flat), unchanged.
  - Removal/clear paths already detach via `parent = None`; empty epoch `Node`s may remain—acceptable unless you later add explicit cleanup.

## Files touched

- Primary: `[pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)` — new helper, new field, signature/threading changes as above.
- No change required to `[vispy_helpers.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py)` unless you later want parent-checkbox propagation.

## Verification

- Run a workflow that calls `add_epoch_visuals` for several epochs; confirm Scene Tree shows `Epoch N` nodes with children (contours per `t`, fills, extrusions, plane) nested underneath.
- Confirm `set_active_epoch` still toggles contour/plane visibility as before.
- Confirm `add_posterior_contours` without `scene_parent` still builds a flat list at the view root.

