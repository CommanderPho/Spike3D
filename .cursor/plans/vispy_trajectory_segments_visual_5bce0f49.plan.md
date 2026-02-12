---
name: Vispy trajectory segments visual
overview: Add a single vispy-based class that renders a List[pd.DataFrame] of 2D trajectory segments on one canvas with configurable per-segment styling, using one Line (with connect array) when styles are uniform for efficiency, or one Line per segment when styles differ.
todos: []
isProject: false
---

# Vispy trajectory segments visual

## Goal

Implement one usable entry point (class or function) that:

- Accepts `**List[pd.DataFrame]**` as segments of a 2D position trajectory.
- Renders all segments on a **single canvas** (one parent Node).
- Lets the user **specify how each segment is drawn** (color, width, method).
- Is **efficient**: prefer a single vispy Line with a connect array when all segments share width/method; otherwise one Line per segment under the same parent.

## Location and reuse

- **File**: Add to [vispy_helpers.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py) next to `create_contour_line_visuals` and `VispyHelpers.create_heading_rainbow_line`, to keep vispy helpers in one place. No new file unless you prefer a dedicated module (e.g. `trajectory_segments_visual.py`) for clarity.
- **Reuse**: Use existing `_color_to_rgba_tuple()` and, if needed, `_colormap_colors` / `_contour_colors_for_masks` for color lists. Use `vz.Line` and `vispy.scene.node.Node` as in the rest of the file. Add `import pandas as pd` at top if not present.

## Data contract

- Each segment is a `**pd.DataFrame**` with at least two columns used for 2D position (default `'x'` and `'y'`).
- Configurable column names via `**x_col='x'`, `y_col='y'**` (or similar) so callers can pass DataFrames with different column names (e.g. `binned_x`/`binned_y`).
- Extract positions as `np.float32` with shape `(n_points, 2)` per segment; skip or drop NaN rows so each segment yields a valid polyline.

## Per-segment “how each are drawn”

Support at least:

- **Global style**: one `color`, one `width`, one `method` applied to all segments (e.g. `line_width=2.0`, `method='gl'`).
- **Per-segment color**: `colors` as a sequence of length `len(segments)` (color spec per segment), or a **callable** `segment_style(segment_index: int, segment_df: pd.DataFrame) -> dict` returning e.g. `{'color': ..., 'width': ..., 'method': 'gl'}` so each segment can be styled differently.

When all segments use the **same** `width` and `method`, the implementation should use a **single** vispy `Line` with:

- **Concatenated positions**: all segment positions stacked into one `(N, 2)` array.
- **Connect array**: `(M, 2)` int array of vertex index pairs, connecting only consecutive vertices **within** each segment (no link between the last vertex of segment i and the first of segment i+1). This yields one draw call and one Line for maximum efficiency.

When `width` or `method` differ per segment, create **one `vz.Line` per segment** (as in `create_contour_line_visuals`) and attach all to the same parent Node; the user still gets “one visual” (one node on the canvas) with full per-segment control.

## API shape (single class)

- **Class**: `TrajectorySegmentsVisual(vispy.scene.node.Node)` (or a plain Node that owns the Line(s)).
  - **Constructor**: e.g.  
  `__init__(self, segments: List[pd.DataFrame], parent: Optional[Node] = None, *, x_col: str = 'x', y_col: str = 'y', color: Optional[Union[str, Tuple, Sequence]] = None, colors: Optional[Sequence] = None, line_width: float = 2.0, line_widths: Optional[Sequence[float]] = None, method: str = 'gl', segment_style: Optional[Callable[[int, pd.DataFrame], dict]] = None, order: int = 10)`  
  Resolve `color`/`colors`/`segment_style` so that per-segment color (and optionally width/method) are well-defined; then choose single-Line vs multi-Line path and create the child visual(s).
  - **Optional**: `set_data(self, segments: List[pd.DataFrame])` to update segments and refresh the Line(s) without creating a new visual (recompute positions and connect array / per-segment lines and call `set_data` or replace children as needed).
  - Expose the underlying visual(s) if useful (e.g. `.line` for single-Line mode, `.lines` for multi-Line mode) so callers can set `set_gl_state('translucent', depth_test=False)` etc. if they want.
- **Usage**: User adds this node to the scene once, e.g.  
`seg_visual = TrajectorySegmentsVisual(segments, parent=view.scene, colors=['r','g','b'], line_width=1.5)`  
and optionally calls `seg_visual.set_data(new_segments)` later.

If you prefer a **single function** instead of a class: a function `create_trajectory_segments_visual(segments, parent, **kwargs)` that builds the same Node and Line(s), attaches to `parent`, and returns the Node is acceptable; the plan above applies with “function that returns the node” instead of a stateful class with `set_data`. The class form is slightly more flexible for updates.

## Implementation details

1. **Position extraction**: For each DataFrame in `segments`, take `df[x_col]` and `df[y_col]`, drop or skip NaN rows, form `np.column_stack([x, y]).astype(np.float32)`. Filter to segments with at least 2 points (a segment of 1 point cannot draw a line).
2. **Connect array (single-Line path)**: For each segment with `n_i` points, if global offset into the concatenated pos array is `off`, add edges `(off, off+1), (off+1, off+2), ..., (off+n_i-2, off+n_i-1)`. Stack into one `(num_edges, 2)` int array and pass to `vz.Line(..., connect=connect)`.
3. **Vertex colors (single-Line path)**: For each segment, assign one RGBA tuple (from `color`, `colors[i]`, or `segment_style(i, df)['color']`); repeat that RGBA for every vertex of that segment to get a `(N, 4)` float32 array for `color=`.
4. **Multi-Line path**: Same position extraction per segment; for each segment create one `vz.Line(pos=pos_i, color=rgba_i, width=width_i, method=method_i, parent=self)` with segment-specific style from `segment_style` or from `colors`/`line_widths` lists. Set `line.order = order` (and optionally different order per segment if needed).
5. **Docstring**: Document that each segment is a 2D trajectory DataFrame (default columns `x`, `y`), and that styling can be global or per-segment via `colors` / `segment_style`; mention the efficiency behavior (single Line when width/method are uniform).

## Testing / example

- Add a short `if __name__ == '__main__'` block or a small example in the docstring that builds 2–3 `pd.DataFrame` segments (e.g. with `'x'` and `'y'`), creates `TrajectorySegmentsVisual(segments, parent=scene, colors=[...])` on a `SceneCanvas` with a panzoom view, and runs `app.run()` so the result can be visually checked. This matches the existing pattern in [vispy_helpers.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py) (e.g. contour example at bottom).

## Summary

- **Single entry point**: One class `TrajectorySegmentsVisual` (or one function `create_trajectory_segments_visual`) in [vispy_helpers.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py).
- **Input**: `List[pd.DataFrame]` with configurable `x_col`/`y_col`.
- **Efficiency**: One Line with `connect` + vertex colors when all segments share width/method; otherwise one Line per segment under one parent.
- **Per-segment styling**: `color`/`colors` and optional `line_width`/`line_widths` or `segment_style(idx, df)`.
- **Conventions**: Follow existing vispy_helpers patterns (single-line signatures where possible, two blank lines between methods, minimal edits), and reuse `_color_to_rgba_tuple` for color parsing.

