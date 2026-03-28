---
name: Turntable MMB pan
overview: "Add a `viewbox_mouse_event` override to `CustomTurntableCamera` that mirrors `CustomArcballCamera`: after delegating to VisPy’s base handler, treat middle-button drag as center translation using the existing `_dist_to_trans` / `_get_dim_vectors` pipeline (same as Shift+LMB)."
todos:
  - id: add-mmb-handler
    content: Add CustomTurntableCamera.viewbox_mouse_event mirroring CustomArcballCamera (82-105)
    status: completed
  - id: docstring
    content: Update CustomTurntableCamera Interaction notes for MMB pan
    status: completed
isProject: false
---

# Middle-mouse pan for CustomTurntableCamera

## Context

- VisPy’s `[Base3DRotationCamera.viewbox_mouse_event](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/.venv/Lib/site-packages/vispy/scene/cameras/perspective.py)` handles **1** = LMB (orbit / Shift+orbit pan), **2** = RMB (zoom / Shift+FOV). It does **not** handle button **3** (middle).
- `[CustomArcballCamera.viewbox_mouse_event](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_cameras.py)` calls `super()`, then on `mouse_move` with MMB down (no modifiers), applies the same translation recipe as the base’s Shift+LMB branch: normalize drag by viewbox size and `_scale_factor`, flip Y, `_dist_to_trans` → world axes via `_get_dim_vectors` and `_flip_factors`, accumulate into `_event_value` as the pan anchor.
- `[CustomTurntableCamera](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_cameras.py)` already defines `_dist_to_trans` for roll/azimuth/elevation and uses the default `_get_dim_vectors()` from the base (unlike arcball’s `np.eye(3)[::-1]` override). Copying the **same** MMB block as arcball is therefore correct and consistent.

## Implementation

**File:** [pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_cameras.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_cameras.py)

1. **Add `viewbox_mouse_event`** on `CustomTurntableCamera` immediately after `__init__` (or before `_update_rotation`), matching `CustomArcballCamera` lines 82–105:
  - `super(CustomTurntableCamera, self).viewbox_mouse_event(event)`
  - Early exits: not `interactive`; not `mouse_move` or no `press_event`; skip if both buttons 1 and 2 (same as base); require button **3** and **no** modifiers.
  - Pan anchor: `if self._event_value is None or len(self._event_value) == 2: self._event_value = self.center` (same heuristic as arcball: distinguishes 3D `center` from 2-tuple rotation state).
  - Compute `dist` from press vs current pos, then identical `dx,dy,dz` chain as arcball / base Shift+LMB.
2. **Docstring:** Under `Notes` → `Interaction`, add one bullet: **MMB drag: translate the center point (same as SHIFT + LMB)**, aligned with the existing `CustomArcballCamera` doc.

## Scope / non-goals

- No changes to VisPy vendored code.
- No refactor to deduplicate the duplicated MMB block between the two classes unless you want that as a follow-up (would be a small private helper in the same module).

## Verification

- Manually: turntable view — MMB drag pans scene center; Shift+LMB still pans; LMB still orbits; RMB still zooms.

