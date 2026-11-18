---
name: animal_local_axes_overlay
overview: Add a compact 3D XYZ axis triad at the animal’s current position in the interactive PyVista browser, updating in lockstep with the existing current-location circle.
todos:
  - id: add-mixin-local-axes-helper
    content: Add a new mixin helper that renders a compact XYZ arrow triad anchored at a provided point and updates by stable mesh name.
    status: completed
  - id: wire-into-animal-update
    content: Call the new helper from perform_update_animal_position_variables() using curr_animal_point so triad and circle move together each frame.
    status: completed
  - id: add-toggle-and-defaults
    content: Add a params flag and sizing defaults to enable/disable and tune the local coordinate triad.
    status: completed
  - id: verify-runtime-behavior
    content: Validate slider-driven updates keep axes attached and that no actor accumulation or camera reset is introduced.
    status: completed
isProject: false
---

# Add Local XYZ Axes At Animal Position

## Goal
Render a small 3D coordinate triad (`+X`, `+Y`, `+Z`) at the current animal position and keep it moving with the same per-frame updates that drive `animal_current_location_point`.

## Files To Change
- [`pyphoplacecellanalysis/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/InteractivePlotterMixins.py`](pyphoplacecellanalysis/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/Mixins/InteractivePlotterMixins.py)
- [`pyphoplacecellanalysis/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/InteractivePlaceCellDataExplorer.py`](pyphoplacecellanalysis/src/pyphoplacecellanalysis/GUI/PyVista/InteractivePlotter/InteractivePlaceCellDataExplorer.py)

## Implementation Approach
1. Add a new plotting helper in `InteractivePyvistaPlotter_PointAndPathPlottingMixin` to build/update a compact axis triad mesh anchored at `curr_animal_point`.
   - Use `pyvista` geometric primitives (`pv.Arrow` for each axis, optionally merged via `pv.append_polydata`) so the triad is represented as one logical mesh update.
   - Keep sizes small by default (e.g., shorter shaft/tip than heading triangles) and expose size kwargs (`axis_length`, `shaft_radius`, `tip_length`, `tip_radius`) for tuning.
   - Add mesh with deterministic names and `reset_camera=False`, matching existing update conventions.

2. Integrate the new helper into the existing animal-position update flow in `InteractivePlaceCellDataExplorer.perform_update_animal_position_variables(...)`.
   - Immediately after updating `animal_current_location_point`, call the new helper using the same `curr_animal_point`.
   - This ensures the axes always move in the same update cycle as the circle and heading/momentum overlays.

3. Add a lightweight feature toggle/config default.
   - Introduce `self.params.setdefault('enable_animal_local_coordinate_axes', True)` (or `False` default if preferred) during setup.
   - Gate rendering call in `perform_update_animal_position_variables(...)` with this flag to preserve backwards compatibility and easy disable.

4. Maintain rendering/performance semantics.
   - Keep `render=False` in helper calls during per-frame updates and rely on existing final `self.p.render()` call.
   - Reuse same mesh names each frame so actors update rather than accumulating.

## Why This Fits Existing Architecture
- The class already centralizes per-frame position visuals in `perform_update_animal_position_variables(...)` (circle + heading + momentum), making this the correct single insertion point.
- Existing mixin methods follow the same add/update-by-name pattern; the new helper aligns with that pattern and minimizes churn.

## Validation
- Open `_display_3d_interactive_spike_and_behavior_browser`, scrub/play slider, confirm the XYZ triad remains centered on the moving location circle.
- Confirm no actor accumulation (constant actor count for the triad) and no camera reset/flicker.
- Verify the toggle disables only the local axes without affecting current circle/heading/momentum visuals.