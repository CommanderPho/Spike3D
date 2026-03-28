---
name: Middle-button arcball pan
overview: Extend `CustomArcballCamera` in pyPhoPlaceCellAnalysis to treat middle-mouse drag as the same center-translation path as `Base3DRotationCamera`’s SHIFT+LMB branch in VisPy’s `perspective.py`, without forking VisPy itself.
todos:
  - id: override-mouse
    content: Add viewbox_mouse_event override + post-super MMB (button 3) branch mirroring perspective.py SHIFT+LMB pan math
    status: completed
  - id: docstring
    content: Update CustomArcballCamera interaction docstring for MMB pan
    status: completed
isProject: false
---

# Middle-mouse pan for CustomArcballCamera

## Reference behavior (VisPy upstream)

`[Base3DRotationCamera.viewbox_mouse_event](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/.venv/Lib/site-packages/vispy/scene/cameras/perspective.py)` (lines 201–269) maps:

- **Button 1 (LMB)**, no modifiers → `_update_rotation` (orbit).
- **Button 2 (RMB)**, no modifiers → zoom via `scale_factor` / `distance`.
- **SHIFT + LMB** → translate `center` using `_dist_to_trans`, `_get_dim_vectors()`, and `_flip_factors` (the “pan” you want to mirror).
- **SHIFT + RMB** → adjust `fov`.

VisPy’s `[MouseEvent](H:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/.venv/Lib/site-packages/vispy/app/canvas.py)` documents **middle = 3**. On `mouse_move` with **only** button 3 pressed, none of the `elif` branches in `Base3DRotationCamera` run, so middle drag is currently a no-op — safe to specialize in the subclass.

## Implementation (single file)

**File:** [pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_cameras.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_cameras.py)

1. **Override `viewbox_mouse_event(self, event)`** on `CustomArcballCamera`:
  - Call `super().viewbox_mouse_event(event)` first so wheel zoom, gesture zoom, press/release handling, and `_event_value` reset on `mouse_release` stay identical to upstream.
  - After `super()`, if `not self.interactive`, return.
  - If `event.type == 'mouse_move'` and `event.press_event is not None`:
    - Apply the **same guard** as the base: skip if both `1` and `2` are in `event.buttons` (line 223–224 in `perspective.py`).
    - If `**3 in event.buttons`** and `**not event.mouse_event.modifiers`** (plain middle drag; matches “middle button” without extra qualifiers):
      - Run the **same body** as the SHIFT+LMB translate block (lines 249–262): `norm`, `_event_value` / `center` bookkeeping, `dist = (p1 - p2) / norm * self._scale_factor` with `dist[1] *= -1`, `_dist_to_trans`, `_get_dim_vectors`, `_flip_factors`, update `self.center`.
       Using **post-`super()`** avoids re-implementing the full method and keeps behavior aligned when VisPy fixes unrelated branches.
2. **Docstring** under `Notes` / `Interaction`: add a bullet that **MMB drag** pans the center (same as SHIFT+LMB).
3. **No changes** to installed `perspective.py` — all logic stays in your package.

## Edge cases (acceptable defaults)

- **SHIFT + MMB**: With `not modifiers`, SHIFT+MMB will not use this branch; user can still SHIFT+LMB to pan. If you later want SHIFT+MMB to pan too, relax the modifier check to allow SHIFT only.
- **Multiple buttons**: Respect the base’s `1 and 2` early return; do not add middle pan when that combo is active.

## Verification

- Manual: use the existing usage snippet in the class docstring (or any view using `CustomArcballCamera`): LMB orbit, RMB zoom, **MMB drag** moves the volume/scene center like SHIFT+LMB.

