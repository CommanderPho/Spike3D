---
name: Fix position line alignment
overview: The white position trajectory in the 3D volumetric viewer is drawn from raw `curr_position_df` x/y with no transform, while the arena, axes, and decoder visuals use the xbin/ybin coordinate frame. Aligning the trajectory to that frame in one place fixes the misalignment with minimal code.
todos: []
isProject: false
---

# Fix position trajectory alignment in 3D volumetric viewer

## Root cause

- **Position line**: Built in `[setup_position_trajectory_curves()](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)` from `curr_position_df['x']` and `['y']` and stacked with time-derived `z_vals` into `pos3d`. No transform is applied.
- **Arena / scene**: Arena wireframe, axes, posterior planes, and contours all use **decoder space**: origin at `(xbin[0], ybin[0])`, extent `[xbin[0], xbin[-1]]` x `[ybin[0], ybin[-1]]`.

If position data lives in a different frame (e.g. different origin, scale, or rotation from the decoder’s bins), the trajectory will appear shifted and/or rotated relative to the blue arena box.

## Recommended fix (easiest): align trajectory to arena in viewer

Apply a **linear map from the trajectory’s XY extent to the arena extent** when building `pos3d`. That corrects shift and scale in one place; optional rotation can be added later if needed.

**File**: [predicitive_decoding_vispy.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py) (class `Volumentric2DTimeSeriesPlotter`).

**Change**: In `setup_position_trajectory_curves()` (around 1926–1958), after computing `x_vals`, `y_vals` and before building `z_vals` / `pos3d`:

1. Compute position extent:
  `x_min_p, x_max_p = x_vals.min(), x_vals.max()`  
   `y_min_p, y_max_p = y_vals.min(), y_vals.max()`
2. Map XY into arena coordinates (avoid div-by-zero with a small epsilon):
  - `x_span_p = max(x_max_p - x_min_p, 1e-6)`, `y_span_p = max(y_max_p - y_min_p, 1e-6)`
  - `x_vals = xbin[0] + (x_vals - x_min_p) / x_span_p * (self.xbin[-1] - self.xbin[0])`
  - `y_vals = ybin[0] + (y_vals - y_min_p) / y_span_p * (self.ybin[-1] - self.ybin[0])`

Then keep the rest of the method as-is (z from time, `pos3d = column_stack((x_vals, y_vals, z_vals))`).

**Effect**: The trajectory’s XY bounding box is scaled and shifted to exactly fill the arena box, so it will align with the grid and contours. If the only issue was offset/scale, this fixes it; if there is a rotation between position and decoder space, the trajectory will still be axis-aligned with the arena (rotation would require an extra optional parameter and 2D rotation step).

## Alternative: translation-only

If you know the mismatch is **only a constant offset** (same scale, different origin):

- `x_vals += (float(self.xbin[0]) - float(np.min(x_vals)))`
- `y_vals += (float(self.ybin[0]) - float(np.min(y_vals)))`

Use this only when position and decoder share the same scale and you just want the bottom-left of the path at the arena’s bottom-left.

## Optional: make alignment configurable

To support pipelines where position is already in decoder space (no transform needed), add a flag, e.g. `align_position_to_arena: bool = True` on `Volumentric2DTimeSeriesPlotter` and in `init_from_position_and_decoder` (and `**kwargs` in `setup()`). When `False`, skip the mapping and keep current behavior. Default `True` gives the fix above for existing callers.

## Call sites (no change required)

- [predicitive_decoding_vispy.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py) (docstring example)
- [vispy_helpers.py](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py) (~525)

Existing calls pass `curr_position_df`, `xbin`, `ybin`; no API change needed unless you add the optional `align_position_to_arena` parameter.

## Summary


| Approach                                                   | Effort                   | Fixes              |
| ---------------------------------------------------------- | ------------------------ | ------------------ |
| **Extent mapping in `setup_position_trajectory_curves()`** | Small edit in one method | Shift + scale      |
| Translation-only                                           | Same place, 2 lines      | Shift only         |
| Optional `align_position_to_arena`                         | Add one field + one `if` | Same, plus opt-out |


Recommended: implement the extent mapping in `setup_position_trajectory_curves()`; add `align_position_to_arena` only if you need to disable it for some callers.