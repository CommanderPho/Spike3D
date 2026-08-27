---
name: Track endcap separator lines
overview: Add two thin dark vertical lines per track shape (long and short) in the track remapping diagram, drawn at the inner endcap/body boundaries. All changes belong in `track_shape_drawing.py`; the display wrapper in `DirectionalPlacefieldGlobalComputationFunctions.py` requires no edits.
todos:
  - id: add-helper
    content: Add `_build_track_1D_endcap_separator_line_segments` helper next to `_build_track_1D_verticies` in track_shape_drawing.py
    status: completed
  - id: extend-helper-return
    content: Compute and return endcap separator line segments from `_plot_helper_add_track_shapes`
    status: completed
  - id: draw-lines
    content: Unpack and draw separator lines in `_plot_track_remapping_diagram` after adding track patches
    status: completed
isProject: false
---

# Add endcap separator lines to track remapping diagram

## Context

The figure shown in your screenshot is produced by this call chain:

```mermaid
flowchart LR
  DisplayFn["_display_directional_track_remapping_diagram"] --> PlotBidir["TrackRemappingDiagramFigure.plot_bidirectional_track_remapping_diagram"]
  PlotBidir --> PlotSingle["_plot_track_remapping_diagram"]
  PlotSingle --> AddShapes["_plot_helper_add_track_shapes"]
  AddShapes --> BuildVerts["_build_track_1D_verticies"]
```

[`DirectionalPlacefieldGlobalComputationFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py) (lines 9896–9956) only delegates to `TrackRemappingDiagramFigure.plot_bidirectional_track_remapping_diagram`. **No changes are needed there.**

Track polygon geometry is built in [`track_shape_drawing.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py):

- `_build_track_1D_verticies` (line ~1222) — builds the stepped track outline with raised endcap platforms
- `_plot_helper_add_track_shapes` (line ~1658) — builds long/short paths, applies y-scale/translate transforms, returns `PathPatch` objects
- `_plot_track_remapping_diagram` (line ~2000) — adds patches to the axis (once per LR/RL subplot)

### Geometry (from `_build_track_1D_verticies`)

Each track has 3 x-segments: `[left endcap | body | right endcap]`.

The **inner endcap edges** (where platforms meet the body) are at:

- Left: `x = platform_length + track_center_offset_x` (22 cm offset from track center)
- Right: `x = (platform_length + track_length) + track_center_offset_x`

Each separator line runs from `(x, platform_1D_height)` down to `(x, 0)` in local coordinates, then through the same `Affine2D` transform already applied to the track path (`scale(1, track_y_height).translate(0, baseline)`).

There is an existing `perform_add_vertical_track_bounds_lines` helper, but it draws **full-axis-height** dashed boundary lines with labels — not the short in-shape dividers you want. Do not reuse it.

## Proposed changes (single file)

**File:** [`track_shape_drawing.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/track_shape_drawing.py)

### 1. Add helper `_build_track_1D_endcap_separator_line_segments` (near `_build_track_1D_verticies`, ~line 1302)

Small function mirroring the x-offset math from `_build_track_1D_verticies`:

```python
def _build_track_1D_endcap_separator_line_segments(platform_length, track_length, platform_1D_height, track_center_midpoint_x, transformation) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    total_track_length = platform_length + track_length + platform_length
    track_center_offset_x = track_center_midpoint_x - total_track_length / 2.0
    left_inner_x = platform_length + track_center_offset_x
    right_inner_x = platform_length + track_length + track_center_offset_x
    local_segments = [
        ((left_inner_x, platform_1D_height), (left_inner_x, 0.0)),
        ((right_inner_x, platform_1D_height), (right_inner_x, 0.0)),
    ]
    # apply `transformation.transform()` to each endpoint
    return transformed_segments
```

### 2. Extend `_plot_helper_add_track_shapes` return value (~lines 1746–1768)

After building `long_path` / `short_path` and their transforms, compute separator segments using the **same** `platform_length=22.0`, `track_length`, `platform_1D_height`, `track_center_midpoint_x`, and `long_transformation` / `short_transformation` already in scope.

Change return from:
```python
return (long_patch, long_path), (short_patch, short_path)
```
to:
```python
return (long_patch, long_path, long_endcap_separator_lines), (short_patch, short_path, short_endcap_separator_lines)
```

Pick line color from `is_dark_mode`: `'#333333'` when light background (default), `'#CCCCCC'` when dark background — consistent with existing `_default_fg_color` logic in `_plot_track_remapping_diagram`.

### 3. Draw lines in `_plot_track_remapping_diagram` (~lines 2000–2014)

Update unpacking and, immediately after `ax.add_patch(long_patch)` / `ax.add_patch(short_patch)`, plot all 4 line segments (2 per track):

```python
for (x0, y0), (x1, y1) in long_endcap_separator_lines + short_endcap_separator_lines:
    ax.plot([x0, x1], [y0, y1], color=endcap_separator_line_color, linewidth=0.75, zorder=2, solid_capstyle='butt')
```

- `zorder=2` — above track fill patches (default 1), below scatter points (zorder 9+)
- `linewidth=0.75` — thin but visible at publication DPI

Optionally store line artists on `extant_plot_container.plots` (e.g. `long_endcap_separator_lines`) for consistency with patch storage; not required for rendering.

## What stays unchanged

- [`DirectionalPlacefieldGlobalComputationFunctions.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py) — no edits
- `plot_bidirectional_track_remapping_diagram` — no signature changes; both LR and RL subplots inherit the lines automatically via `_plot_track_remapping_diagram`
- `_build_track_1D_verticies` — unchanged; separator math is derived from the same parameters

## Verification

Re-run the track remapping display for a session with LR/RL data and confirm:
- 4 tracks total (2 per subplot) each show **2 vertical lines** at endcap/body junctions
- Lines start at the inner top corner of each endcap platform and reach the track baseline
- Lines do not obscure scatter points or remapping arrows
- Both long (red) and short (blue) tracks show separators at correct x positions (~22 cm inset from each end)
