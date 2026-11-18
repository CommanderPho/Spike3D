---
name: Volumentric2D Plotter
overview: Implement `Volumentric2DTimeSeriesPlotter` as a minimal vispy-backed 3D viewer that plots an animal's 2D open-field trajectory (x, y) over time (z), overlays decoded posterior slices as image planes, and supports highlighted time ranges.
todos:
  - id: impl-class-skeleton
    content: Add @define attrs class skeleton with all data/UI fields below the existing stub
    status: completed
  - id: impl-setup
    content: Implement setup() — derive t_min/t_max, z_scale, pos3d
    status: completed
  - id: impl-buildUI
    content: Implement buildUI() — Qt window, 3D canvas, TurntableCamera, position line, bounding box, slider, key events
    status: completed
  - id: impl-posterior-plane
    content: Implement _build_posterior_plane(t_bin_idx) and update_active_t_bin(t_bin_idx)
    status: completed
  - id: impl-highlights
    content: Implement _build_highlight_bands() for colored time-range highlight bands and labels
    status: completed
  - id: impl-events
    content: Implement on_key_press and on_slider_value_changed handlers
    status: completed
  - id: impl-classmethod
    content: Implement init_from_position_and_decoder classmethod
    status: completed
isProject: false
---

# Implement `Volumentric2DTimeSeriesPlotter`

## Target File

`[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` — append implementation after the existing stub at line 1699.

## Class Structure

Follow the same `@define(slots=False, repr=False, eq=False)` attrs pattern used by `PredictiveDecodingVispyWidget`.

### Data fields

- `curr_position_df: pd.DataFrame` — must have `t`, `x`, `y` columns
- `xbin: np.ndarray`, `ybin: np.ndarray` — arena spatial bin edges
- `p_x_given_n: Optional[np.ndarray]` — decoded posteriors, shape `(n_xbins, n_ybins, n_tbins)`, optional
- `t_bin_edges: Optional[np.ndarray]` — time bin edge array, optional
- `highlight_epochs: Optional[pd.DataFrame]` — DataFrame with `start`, `stop`, optional `color` columns
- `active_t_bin_idx: int = 0`

### Derived fields (populated in `setup()`)

- `t_min`, `t_max`: float, recording time bounds
- `z_scale`: float, scaling factor so time axis is visually proportional

### UI fields (populated in `buildUI()`)

- `canvas`, `main_window`, `view` (3D ViewBox)
- `position_line`: `vz.Line` visual — the 3D trajectory
- `posterior_plane`: `vz.Image` visual — posterior slice at active time bin
- `highlight_boxes: List` — colored transparent highlight band visuals
- `highlight_labels: List` — `vz.Text` labels on x=0, y=0 planes
- `t_bin_slider`, `t_bin_value_label`

## 3D Scene Layout

```
x-axis = arena x position  (matches xbin extent)
y-axis = arena y position  (matches ybin extent)
z-axis = time              (t_min → t_max, scaled by z_scale)
camera = scene.TurntableCamera (elevation=30, azimuth=135)
```

## Key Methods

### `setup()`

- Derive `t_min`, `t_max` from `curr_position_df['t']`
- Compute `z_scale = (xbin[-1]-xbin[0]) / max(t_max-t_min, 1e-6)` so time axis matches arena width
- Build `pos3d`: `(N, 3)` array `[x, y, (t - t_min) * z_scale]` from position df

### `buildUI()`

- Create `scene.SceneCanvas(keys='interactive', show=False, ...)`
- Wrap in `QMainWindow` with `canvas.native` + a `QSlider` for `t_bin_idx` (same pattern as `PredictiveDecodingVispyWidget.buildUI()`)
- Add a single `view = canvas.central_widget.add_view()` with `TurntableCamera`
- Draw arena bounding box as a `vz.Line` wireframe at z=0 and z=total_duration
- Draw position line via `vz.Line(pos=pos3d, color='white', width=2, parent=view.scene)`
- If `p_x_given_n` provided: call `_build_posterior_plane()` for `active_t_bin_idx`
- If `highlight_epochs` provided: call `_build_highlight_bands()`
- Connect `canvas.events.key_press` → `on_key_press`
- Connect slider `valueChanged` → `on_slider_value_changed`

### `_build_posterior_plane(t_bin_idx)`

- Extract slice `img = p_x_given_n[:, :, t_bin_idx].T` (shape `n_ybins × n_xbins`)
- Normalize to `[0, 1]`
- Create RGBA using a colormap (vispy `Colormap('hot')`)
- Create `vz.Image(img_rgba, parent=view.scene)`
- Apply `scene.transforms.MatrixTransform` to scale to arena extent and translate to `z = (t_bin_center - t_min) * z_scale`
- Store in `self.posterior_plane`

### `update_active_t_bin(t_bin_idx)`

- Remove old `self.posterior_plane` from parent, rebuild via `_build_posterior_plane(t_bin_idx)`
- Update `self.active_t_bin_idx`

### `_build_highlight_bands()`

- For each row in `highlight_epochs`, build a semi-transparent colored rectangular prism (6-face mesh or 4 `Line` verticals) spanning `x: [xbin[0], xbin[-1]]`, `y: [ybin[0], ybin[-1]]` at `z: [(row.start - t_min)*z_scale, (row.stop - t_min)*z_scale]`
- Use `vz.Mesh` with `face_colors` at `alpha=0.15` for the band, plus a `vz.Text` label

### `on_key_press(event)`

- Left/Right arrows decrement/increment `active_t_bin_idx` (clamped), call `update_active_t_bin`

### `on_slider_value_changed(value)`

- Call `update_active_t_bin(value)`

### Classmethod `init_from_position_and_decoder(cls, curr_position_df, xbin, ybin, p_x_given_n, t_bin_edges, highlight_epochs=None, **kwargs)`

- Thin convenience constructor

## Vispy Transform for Posterior Plane

```python
n_y, n_x = img_rgba.shape[:2]
x_scale = (xbin[-1] - xbin[0]) / n_x
y_scale = (ybin[-1] - ybin[0]) / n_y
z_val   = (t_bin_center - t_min) * z_scale
t = scene.transforms.MatrixTransform()
t.scale((x_scale, y_scale, 1.0))
t.translate((xbin[0], ybin[0], z_val))
posterior_plane.transform = t
```

