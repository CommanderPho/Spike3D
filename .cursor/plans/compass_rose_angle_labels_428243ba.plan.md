---
name: Compass rose angle labels
overview: Add eight small degree labels on `HeadingCompassRoseVisual` using one batched `scene.visuals.Text` (list of strings + positions), with headings where 0° is North/up and East is 90°, matching the module docstring—not the line-hue mapping helper, which uses a different convention for colors only.
todos:
  - id: label-builder
    content: Add build_heading_compass_rose_label_data (8 strings + positions, display_deg = (90 - math_deg) % 360)
    status: completed
  - id: wire-text
    content: Instantiate scene.visuals.Text in HeadingCompassRoseVisual.__init__ with show_labels / font_size / label_pad kwargs
    status: completed
  - id: smoke-demo
    content: Run CompassDemo and verify eight labels around rose
    status: completed
isProject: false
---

# Compass rose degree labels (0° = North/up)

## Context

- Rose geometry is built in `[position_heading_angle.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` with eight directions: `angles = np.linspace(0, 2 * np.pi, 9)[:-1]`, alternating major/minor spoke lengths.
- VisPy supports **one** `scene.visuals.Text` with `text=[...]` and `pos` as an `(N, 2)` array ([TextVisual API](https://github.com/vispy/vispy/blob/main/vispy/visuals/text/text.py): list of strings + matching positions). That is the efficient approach (single visual, one font atlas), versus eight separate Text nodes.
- **Displayed heading** (per module docstring and `[CompassDemo](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` printout): clockwise compass bearing from North, with **North/up = 0°**, **East = 90°**, etc. For a unit direction from `angle` in the rose loop, `atan2`-style degrees are `math_deg = (np.degrees(angle) % 360.0)`; the label value is `**display_deg = (90.0 - math_deg) % 360.0`**. This matches “0° North, 90° East” and yields 0, 45, …, 315 for the eight spokes.
  - Note: `[HeadingAngleHelpers._heading_deg_to_compass_deg](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\plotting\heading_angle_helpers.py)` uses `(headings_deg - 90) % 360` **for vertex colors only**; it is not the same as the docstring’s compass labels. Do **not** reuse it for the numeric label text if you want 90° at East.

## Implementation steps

1. **Add a small classmethod** on `HeadingCompassRoseVisual`, e.g. `build_heading_compass_rose_label_data(...)`, that mirrors the spoke loop in `build_heading_compass_rose_line_data` (same `angles`, same `major_length` / `minor_length` rule) and returns:
  - `label_texts`: list of 8 strings, e.g. `f"{int(round(display_deg))}°"` (Unicode degree).
  - `label_pos`: `(8, 2)` float32 array of **local** positions: unit direction along each spoke × `(spoke_length + margin)` so text sits just outside the line tip. Use a parameter like `label_pad: float = 0.12` (in the same local units as `major_length`) so labels clear the spokes when scaled.
2. **In `__init__`**, after creating `self.line`:
  - Add optional kwargs: `show_labels: bool = True`, `label_font_size: float = 9` (or similar “small” default), `label_pad`, `label_color` (default something readable, e.g. `'w'` or `(1,1,1,0.9)`), reusing patterns from `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` (`scene.visuals.Text` / `vz.Text`).
  - If `show_labels`: `self.labels = scene.visuals.Text(text=..., pos=..., font_size=..., color=..., anchor_x='center', anchor_y='center', parent=self, depth_test=False)` and match line overlay behavior (`depth_test=False`).
  - Store `self.labels = None` when disabled for a clear attribute.
3. **No change required** to parent `STTransform`: labels are children of the same `scene.Node`, so they move/scale with the rose. `[CompassLegendItem](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)` keeps working without edits unless you later want label positions reflected in `_data_dict` (optional).
4. **Manual sanity check**: run `CompassDemo` (choice 2 in `__main__`) and confirm eight labels 0°, 45°, … at North, NE, … with legible size at default scale.

## Files touched

- Only `[position_heading_angle.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\position_heading_angle.py)`: new classmethod, `__init__` parameters, single `Text` visual.

