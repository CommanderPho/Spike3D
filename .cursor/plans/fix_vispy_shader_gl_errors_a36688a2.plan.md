---
name: Fix vispy shader GL errors
overview: Fix the GL_VERTEX_SHADER "syntax error, unexpected $end at token &lt;EOF&gt;" in position_heading_angle.py by addressing empty/truncated shader source (likely due to context timing or shader string handling) and optionally offering a scene.Line-based fallback.
todos: []
isProject: false
---

# Fix VisPy heading-colored line shader errors

## Problem

Running [position_heading_angle.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py) (options 1 or 2) fails with:

```text
RuntimeError: Shader compilation error in GL_VERTEX_SHADER:
    (0) : error C0000: syntax error, unexpected $end at token "<EOF>"
```

The `(0)` and "unexpected $end" / `<EOF>` indicate the **driver is receiving an empty or truncated shader string** (common on Windows/Intel when the source is wrong or not in scope). The same vertex shader is used by `HeadingColoredLine` and `CompassDemo`; `InteractiveHeadingLine` uses the same program.

## Root cause (likely)

- **Context/timing**: The script builds `gloo.Program(VERT_SHADER, FRAG_SHADER)` in `__init__` before the canvas is shown. The GLIR queue is only merged with the canvas context in `program.draw()`. On some backends (e.g. PyQt5 on Windows), creating the program before the context is fully ready can lead to shader source being lost or not applied correctly when the DATA commands are first flushed.
- **Mitigation**: Create the Program when the GL context is already current (e.g. in `on_initialize` or on first `on_draw`), matching the pattern used in working examples (e.g. [quad.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\LibrariesExamples\vispy\examples\gloo\gl\quad.py) which uses `on_initialize` for GL setup).

## Recommended fix (primary)

1. **Defer Program (and buffers/state) creation to `on_initialize**`
  - In `HeadingColoredLine`, `CompassDemo`, and `InteractiveHeadingLine`: move creation of `self.program`, assignment of attributes/uniforms, and `gloo.set_state(...)` from `__init__` into `on_initialize`.
  - In `__init__`: only set canvas size, store any data needed for the line (positions/tangents or builder logic), and call `self.show()`.
  - Implement `on_initialize(self, event)` and create the Program there; then in `on_draw` use the program as now (no draw if program not yet created in the interactive case).
  - This ensures the Program and its shaders are created when the context is current and the backend is ready, so DATA commands see a valid context and the driver receives the full shader source.
2. **Optional shader string tweaks** (if the above is not enough)
  - Omit explicit `#version 120` and let vispy’s `_convert_desktop_shader` add it (avoids any version-line handling quirks).
  - Ensure shader strings are plain str with no BOM; use `.strip()` when building if needed (no change to logic, just hygiene).

## Alternative: use scene.visuals.Line (no custom gloo shaders)

If the driver continues to receive empty shader source even after deferring creation:

- Refactor the demos to use `vispy.scene.SceneCanvas` and `vispy.scene.visuals.Line` with **precomputed per-vertex colors** (heading → HSV → RGB on the CPU), as in [vispy_helpers.create_heading_rainbow_line](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py) (e.g. around line 355).
- This avoids custom gloo vertex/fragment shaders entirely and uses vispy’s built-in Line shaders, which are already tested on the same backend.

## Files to change

- [position_heading_angle.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py): add `on_initialize` and move Program creation (and related state) there for all three classes; keep `on_draw`/`on_resize` logic, using a flag or `self.program` being `None` where needed (e.g. interactive demo).

## Implementation notes

- **HeadingColoredLine**: Build positions/tangents in `__init__`; in `on_initialize` create `self.program`, set attributes/uniforms, `update_projection()`, and `gloo.set_state(...)`.
- **CompassDemo**: Same pattern: build positions/tangents in `__init__`; create program and state in `on_initialize`; keep print of compass legend where it is (after `show()`).
- **InteractiveHeadingLine**: In `__init__` do not create the program; in `on_initialize` create an empty/minimal program (e.g. same shaders but empty buffers) or create the program on first draw when `len(self.positions) >= 2`. Easiest: create program in `on_initialize` with a single dummy vertex so the program exists; then `update_line()` replaces buffers as the user draws.
- **on_draw**: For the interactive case, only call `self.program.draw(...)` if the program exists and has at least 2 points (current logic already guards with `len(self.positions) >= 2`; also check `self.program` is not None if creation is deferred to first draw).
- Preserve existing behavior: same ortho projection, same blend/line state, and same resize/draw logic; only the **time** at which the Program is created changes.

## Summary


| Approach | Action                                                                                                                                            |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Primary  | Defer `gloo.Program` (and related GL state) creation to `on_initialize` in all three classes so shaders are compiled when the context is current. |
| Fallback | If the error persists, refactor to use `scene.SceneCanvas` + `scene.visuals.Line` with CPU-side heading→color and no custom gloo shaders.         |


No changes to vispy’s installed package are required; all edits are in `position_heading_angle.py`.