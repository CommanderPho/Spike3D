---
name: Heading-angle line in Matplotlib
overview: Implement heading-angle shaded trajectory line in the existing Matplotlib helper `_helper_add_gradient_angle_visualizing_line` by reusing the Vispy heading/rainbow color logic and coloring LineCollection segments by segment direction (0°=red, ROYGBIV, 359°≈violet).
todos: []
isProject: false
---

# Heading-angle shaded trajectory line in Matplotlib

## Goal

Make `[_helper_add_gradient_angle_visualizing_line](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py)` in the decoder mixin draw a trajectory line colored by **heading angle** (same semantics as the Vispy implementation in `[create_heading_rainbow_line](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py)`), instead of the current time-based gradient.

## Reference behavior (Vispy)

- [vispy_helpers.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py): **headings_from_positions(pos)** (lines 297–308) computes segment directions in [0, 360) from (N,2) positions; **heading_deg_to_compass_deg** (313–315) converts atan2 (0=East) to compass (0=North); **heading_angles_to_rainbow_colors** (285–293) maps [0, 360) to (N,4) RGBA via HSV (h = angle/360, S=1, V=1). **positions_to_vertex_colors** (318–322) composes these for per-vertex colors (North = Red, ROYGBIV).

## Implementation approach

1. **Reuse heading/color helpers**
  Import from the existing Vispy helpers so behavior stays identical:
  - `headings_from_positions`, `_heading_deg_to_compass_deg`, `heading_angles_to_rainbow_colors` from `pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers`.
2. **Segment-level colors for the line**
  - Build `pos` as (N, 2) from `x`, `y`.  
  - Compute segment directions: `d = np.diff(pos, axis=0)`, then `angle_deg = (np.degrees(np.arctan2(d[:,1], d[:,0])) + 360) % 360` (length N-1).  
  - Convert to compass: `compass_deg = _heading_deg_to_compass_deg(angle_deg)`.  
  - Map to rainbow RGBA: `segment_colors = heading_angles_to_rainbow_colors(compass_deg, alpha=...)` → shape (N-1, 4).  
  - Build segments as today: `points = np.array([x,y]).T.reshape(-1,1,2)`, `segments = np.concatenate([points[:-1], points[1:]], axis=1)`.  
  - Use **LineCollection(segments, colors=segment_colors, ...)** (no `cmap`/`norm`/`set_array`). Apply same linewidth/alpha as current code (or from `LineCollection_kwargs`).
3. **Optional markers**
  When `add_markers=True`, color markers by **per-vertex** heading so they match the line semantics:  
  - Use **positions_to_vertex_colors(pos)** to get (N, 4) RGBA.  
  - `ax.scatter(x, y, s=s, c=vertex_colors, marker='D')`.  
   (Matplotlib scatter accepts (N, 4) RGBA for `c`.)
4. **Signature and parameters**
  - Keep signature `(cls, ax, t, x, y, add_markers=False, s=20.0, time_cmap='viridis', **LineCollection_kwargs)` for compatibility.  
  - Drop use of `t` and `time_cmap` for coloring (they become unused; docstring will state that the line is colored by heading). Optionally in a follow-up, add a `headings_deg: Optional[NDArray] = None` argument to allow precomputed headings (mirroring Vispy) when provided.  
  - Keep the same return type: `(line, _out_markers or None)`.
5. **Docstring**
  - Update the docstring to describe heading-angle coloring (0°=red, ROYGBIV, North=Red) and the optional markers; fix the example to call `_helper_add_gradient_angle_visualizing_line` instead of `_helper_add_gradient_line`.
6. **Edge cases**
  - If `len(x) < 2`, no segments: skip LineCollection (or return early); no markers or return empty/None as appropriate.  
  - Single segment (2 points): one segment color from that segment’s direction.

## Files to change

- [decoder_plotting_mixins.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py): add import for the three helpers from `vispy_helpers`; replace the body of `_helper_add_gradient_angle_visualizing_line` (lines 2277–2301) with heading-based segment colors and optional vertex-colored markers; update docstring.

## Dependency note

Importing from `pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers` will load the vispy package when this code path runs. The mixin already imports from `Pho2D` (e.g. `track_shape_drawing`). If a future requirement is to use this helper in a strictly vispy-free environment, the three helpers could be moved to a shared math/color module (e.g. under `Pho2D`) and used by both vispy_helpers and this mixin; not in scope for this change.