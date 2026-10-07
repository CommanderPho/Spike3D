---
name: Radon debugger overlays
overview: Update RadonTransformDebugger so the existing seconds/centimeters posterior shows the geometric radon line, the vertical neighbor window that enters the score, and the rho/phi normal, instead of the incorrect BandROI.
todos:
  - id: band-mask
    content: Add the vertical scoring-window image overlay in the posterior's real-space origin and scale
    status: completed
  - id: line-rho
    content: Draw active_debug_info.y_line in blue and the rho/phi normal converted to seconds and centimeters
    status: completed
  - id: remove-bandroi
    content: Stop installing the incorrect default BandROI and refresh overlays from the epoch geometry
    status: completed
isProject: false
---

# Draw the real radon line and scoring window

The debugger in [`RadonTransformDebuggerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py) keeps the viridis posterior in seconds and centimeters. Three overlays replace the dotted index curve and the `BandROI` whose width is the integer `n_neighbours`.

The score in [`decoders.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\analyses\decoders.py) does two things, and the plot should show those:

- The geometric line stored on `active_debug_info.y_line` (internal slope times `t`, plus intercept). The velocity returned by `radon_transform` is negated, so `active_radon_values.velocity * t + intercept` is the mirrored line and must not be drawn.
- A vertical window on each in-bounds time column: rows `best_y_line_idxs[ci] ± n_neighbours`, clipped to the position axis. That is the `2 * n_neighbours + 1` box convolved along position before the line is sampled. Columns whose rounded index falls outside the matrix use the column median and get no highlighted cells.

Rho/phi use the same index normal form as `y_line_idxs`: `(ci - ci_mid) * cos(phi) + (ri - ri_mid) * sin(phi) = rho`. The segment runs from `(ci_mid, ri_mid)` to the foot `(ci_mid + rho*cos(phi), ri_mid + rho*sin(phi))`, converted with the debug mapping `t = ci * dt + t[0]` and `x = ri * dx + pos[0]` (bin centers). On this centimeter/second plot the on-screen angle is not phi; the foot is still the index-space foot.

```mermaid
flowchart LR
  posterior[Posterior image cm and sec]
  band[Vertical scoring window]
  line[y_line vs t]
  rho[Rho segment from matrix center]
  posterior --> band --> line --> rho
```

## Drawing

Add three methods and call them from `build_GUI` after `add_real_space_posterior`. Reuse that image’s origin `(time_bin_edges[0], xbin[0])` and scale `(time_bin_size, pos_bin_size)`.

- **Scoring band.** Build a `(n_pos, n_t)` mask: 1 inside the clipped window, NaN elsewhere, only where the rounded line index is inside `[0, n_pos)`. `addImage` it with the same origin and scale, a yellow colormap, and partial alpha so the posterior stays visible. `resetzoom=False`.
- **Line.** Change `add_real_space_curve` to plot `active_debug_info.t` against `active_debug_info.y_line` as a solid blue curve, legend `y(t)=velocity*t+intercept`. Drop the dotted extrapolated `best_y_line_idxs` curve.
- **Rho/phi.** Add a dashed segment between the converted center and foot, with a marker labeled with the numeric `best_rho` and `best_phi`.

`build_GUI` stops creating the `BandROI`, stops `setRois` / `addItem` for it, and still calls `setStats` so a manually drawn ROI can use the existing stat functions. The default view no longer shows a perpendicular band that contradicts the mask.

## Geometry stored for the epoch

In `update_epoch_idx`, replace the block that sets `band_width = float(num_neighbours)` with the geometric endpoints `[t[0], y_line[0]]` and `[t[-1], y_line[-1]]`. Store `band_width` as the vertical window height `(2 * n_neighbours + 1) * pos_bin_size` in centimeters. Do not pass that width to a `BandROI`.

Point `_perform_update_band_ROI`, `update_ROI`, and `on_set_active_epoch_idx_changed` at one refresh that redraws the posterior, band, line, and rho/phi from `active_radon_values` when `window` already exists. `update_GUI` already clears and calls `build_GUI`.

No notebook edits. The open cell only calls `build_GUI()`.
