---
name: 3D Posterior Contour Rendering
overview: Update the 3D posterior contour plan to support filling contours with the same color as the contour line and a default fill alpha of 0.25, using 3D meshes for fill in the volumetric plotter.
todos: []
isProject: false
---

# 3D Posterior Contour Rendering — Contour Fill Support

## Goal

Support **filling** each 3D contour with the same RGB color as its contour line, using a default fill alpha of **0.25**, in addition to the existing line rendering.

## Current state

- [predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py): `Volumentric2DTimeSeriesPlotter` already has `_build_posterior_contours_3d` (lines 2069–2100) and the public API (`add_posterior_contours`, get/list/set_visibility/remove/clear). Only **lines** are created; the plan previously stated that fill is skipped in 3D because `vz.Polygon` is 2D-only.
- 2D reference: [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py) `create_contour_line_visuals` (lines 281–295) uses fill color = line RGB with a separate `fill_alpha` (default 0.3) and `vz.Polygon` for 2D.

## Design

- **Fill color**: Same RGB as the contour line; alpha = configurable fill alpha (default **0.25**).
- **3D fill**: Use **vispy Mesh** (`vz.Mesh`) with vertices = closed contour in 3D `(x, y, z)` and triangulated faces. Each contour is a flat polygon at constant `z`; triangulate the 2D polygon then use the same `z` for all vertices.
- **Ordering**: Draw fill meshes **behind** lines (e.g. fill `order = 21`, line `order = 22` in `_build_posterior_contours_3d`).
- **API**: Add optional args `fill: bool = True` and `fill_alpha: float = 0.25` to `_build_posterior_contours_3d` and `add_posterior_contours`; when `fill=True`, build one Mesh per contour and store them alongside line visuals.

## Implementation outline

**1. Triangulation for 3D fill**

- Reuse closed-contour logic: ensure contour is closed (e.g. same as [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py) `_ensure_closed_pos` — append first point if first and last differ).
- Triangulate the closed 2D polygon. Options:
  - **Simple triangle fan** from first vertex: faces `(0, i, i+1)` for `i = 1 .. n-2`. Works for simple convex/star contours; may overlap for concave.
  - **Robust option**: use `vispy.geometry.polygon.PolygonData` / `vispy.geometry.triangulation` if available, or a small helper that triangulates 2D vertices and returns faces.
- Build 3D vertices: `pos_3d` already used for the line (closed or not); for the mesh use closed `pos_3d` and the same triangulation (faces index into the closed vertex list).

**2. Changes in [predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)**

- `**_build_posterior_contours_3d`**  
  - Add parameters: `fill: bool = True`, `fill_alpha: float = 0.25`.  
  - For each contour item, after building `pos_3d` and the line:
    - If `fill` and `len(pos_3d) >= 3`: close the contour (e.g. reuse or mirror `_ensure_closed_pos` on the 2D part then re-lift to 3D, or close in 3D), triangulate to get `vertices` (N, 3) and `faces` (M, 3). Create `vz.Mesh(vertices=..., faces=..., color=fill_rgba, ...)`, set `mesh.order = 21`, parent to `self.view.scene`. Append to a `fill_visuals` list.
  - Return both line and fill visuals (e.g. `Tuple[List[Any], List[Any]]` or a single flat list with consistent ordering so fills are cleared with lines).
- **Storage and API**  
  - Store both line and fill visuals per key: e.g. `posterior_contours_by_key[identifier] = {'unique_identifier': identifier, 'line_visuals': [...], 'fill_visuals': [...], 'visible': bool}`.  
  - `**add_posterior_contours`**: add kwargs `fill: bool = True`, `fill_alpha: float = 0.25`; pass through to `_build_posterior_contours_3d`; when storing, set visibility on both line and fill visuals.  
  - `**set_posterior_contours_visibility`**: set `visible` on every item in both `line_visuals` and `fill_visuals`.  
  - `**remove_posterior_contours**` and `**clear_posterior_contours**`: detach and clear both `line_visuals` and `fill_visuals` (set `parent = None` for each).

**3. Optional shared helper**

- If desired, add a small helper in [vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py), e.g. `_triangulate_polygon_2d(pos_2d) -> faces` or `create_contour_fill_mesh_3d(pos_3d_closed, rgba, parent, order)` that closes, triangulates, and returns a `vz.Mesh`, to keep triangulation logic in one place and avoid duplicating `_ensure_closed_pos` in the vispy module. Otherwise, implement closing + triangulation inline in `_build_posterior_contours_3d`.

## Summary


| Item       | Detail                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| Fill color | Same RGB as contour line; alpha = `fill_alpha` (default **0.25**)                                                          |
| Default    | Fill on by default (`fill=True`, `fill_alpha=0.25`)                                                                        |
| 3D fill    | `vz.Mesh` with closed contour vertices (3D) and triangulated faces                                                         |
| Draw order | Fill meshes behind lines (e.g. order 21 vs 22)                                                                             |
| API        | `_build_posterior_contours_3d(..., fill=True, fill_alpha=0.25)`; `add_posterior_contours(..., fill=True, fill_alpha=0.25)` |
| Storage    | Per-key dict holds both `line_visuals` and `fill_visuals`; visibility/remove/clear operate on both                         |


No changes to the existing 2D central view pipeline; only the 3D volumetric plotter gains this behavior.