---
name: vispy overlay text helper
overview: Add a `VispyHelpers` classmethod that creates viewport-anchored overlay text in the top-left corner by default, with a fixed message that remains stationary while panning/zooming the camera.
todos:
  - id: add-overlay-helper
    content: Add `VispyHelpers.create_viewport_overlay_text(...)` classmethod with top-left defaults and fixed-message behavior.
    status: completed
  - id: bind-resize-positioning
    content: Anchor text in viewport pixel coordinates and update position on canvas resize to maintain top-left placement.
    status: completed
  - id: add-usage-example
    content: Add/update an example in `vispy_helpers.py` showing overlay text remains stationary during pan/zoom.
    status: completed
isProject: false
---

# Add Viewport Overlay Text Helper

## Goal

Implement a new helper in `[h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)` that draws fixed text in viewport space (top-left by default), independent of camera transforms.

## Planned Changes

- Add a new `@classmethod` on `VispyHelpers` (near other rendering helpers) with a single-line signature, e.g. `create_viewport_overlay_text(...)`, that:
  - accepts a `canvas` (or equivalent viewport root),
  - uses a default message string,
  - supports optional text style params (color/font_size/bold/anchor/margin/order),
  - returns the created text visual so callers can update/remove it later.
- Implement text in canvas/overlay coordinates rather than `view.scene` world coordinates so panning/zooming does not move it.
- Set default placement to top-left with margin offsets from viewport bounds.
- Add a resize callback helper (or lightweight closure) that recomputes text pixel position when canvas size changes, preserving top-left anchoring.
- Keep implementation minimal and additive (no broad refactors), matching the existing utility style in `VispyHelpers`.

## Validation

- Add or extend a small in-file example function (same file) to instantiate a panzoom view and overlay text, then pan/zoom to verify text remains fixed.
- Verify defaults: text appears in top-left corner without extra arguments.
- Verify return object is usable by caller (e.g., mutating `text` content after creation).

## Key Existing Context

- `VispyHelpers` utility methods are grouped in the `VispyHelpers` class and return created visuals where appropriate.
- Existing camera usage relies on pan/zoom in examples, so overlay helper should be canvas-space to avoid camera motion coupling.

