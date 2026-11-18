---
name: Reusable compass rose visual
overview: Deduplicate compass-rose geometry by a shared builder, introduce a `scene.Node`-based visual whose child `Line` lives in unit/local space, and drive placement/size exclusively via `STTransform`. Refactor `CompassDemo` and `CompassLegendItem` to use it while keeping existing public APIs stable where possible.
todos:
  - id: extract-builder
    content: Add build_heading_compass_rose_line_data() with unit geometry + NaN breaks + HeadingAngleHelpers colors
    status: completed
  - id: node-visual
    content: Implement HeadingCompassRoseVisual(scene.Node) with child Line and public .line
    status: completed
  - id: refactor-demo-legend
    content: Refactor CompassDemo and CompassLegendItem to use visual + STTransform; preserve _data_dict world pos via transform.map
    status: completed
isProject: false
---

# Reusable heading compass rose VisPy visual

## Context

- `[position_heading_angle.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` already contains the same cardinal-rose logic twice: `[CompassLegendItem](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` (lines 84–137) and `[CompassDemo](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` (lines 149–173).
- The codebase already uses `**scene.Node` + `STTransform**` for grouped transforms (e.g. `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)`, `[predictive_decoding_central_view.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predictive_decoding_central_view.py)`).

## Design

1. `**build_heading_compass_rose_line_data**` (module-level function, or a small private helper next to the classes)
  - Parameters: `line_points` (default 20), optional `major_length=1.0` / `minor_length=0.5` (same ratio as today: even indices long, odd short).
  - Build the rose **centered at the origin** in 2D (no baked-in `center` or `length` in the vertex array).
  - Return `(positions float32, colors float32, tangents, angle_deg, compass_deg)` mirroring what `CompassLegendItem` currently stuffs into `_data_dict`, so downstream metadata stays available without recomputing.
2. `**HeadingCompassRoseVisual(scene.Node)`** (name can be adjusted; this reads clearly next to `AngleColoredLineVisual`)
  - `__init__(self, parent=None, line_width=2.0, line_points=20, **kwargs)` → call `scene.Node.__init__(self, parent=parent, **kwargs)`, `unfreeze()`, create `**scene.visuals.Line`** as **child of `self`** with `pos`/`color` from the builder, `set_gl_state('translucent', depth_test=False)`, attach `self.line` (public), `freeze()`.
  - Default transform: identity (`NullTransform` / VisPy default) so geometry is unit-sized; callers set `**self.transform = scene.transforms.STTransform(scale=(sx, sy), translate=(tx, ty))**` to place and scale on any `[ViewBox](https://vispy.org/)`.scene.
  - Document in the class docstring (plain pasteable example per project rules): parent to `view.scene`, then assign `transform`.
3. **Refactor `CompassDemo`**
  - Keep canvas/view setup and printout; replace inline loop with `HeadingCompassRoseVisual(parent=self._view.scene, line_width=5.0)` and e.g. `rose.transform = scene.transforms.STTransform(scale=(0.6, 0.6))` to match the previous world extent (major spoke 0.6).
4. **Refactor `CompassLegendItem`**
  - Construct `HeadingCompassRoseVisual(parent=view.scene, line_width=line_width, line_points=line_points)`.
  - Apply `STTransform(scale=(length, length), translate=(float(center[0]), float(center[1])))` so behavior matches today’s baked `center`/`length`.
  - `**self.line**`: point to `rose.line` for backward compatibility.
  - `**_data_dict**`: `[colored_line_advnaced.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\examples\pho_custom\colored_line_advnaced.py)` uses `legend_out._data_dict['pos']` with `VispyHelpers.set_view_camera`. Populate `pos` (and related arrays) as **world/scene coordinates** by mapping the local `positions` through the same `STTransform` (VisPy’s transform `.map` on the vertex array), so camera fitting does not break.
5. **Imports**
  - Add `from vispy.visuals.transforms import STTransform` (or `scene.transforms.STTransform` — pick one style; `vispy_helpers.py` uses `vispy.visuals.transforms`).

## Notes / non-goals

- **Line width** stays in pixels (VisPy default); only vertex positions scale with `STTransform`. This matches typical overlay legends.
- No new markdown files; edits confined to `[position_heading_angle.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` unless you explicitly want the Spike3D example updated to import the new class.

