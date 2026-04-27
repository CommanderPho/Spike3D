---
name: y-up to z-up quaternion remap
overview: Apply a Y-up to Z-up quaternion change-of-basis inside `QuaternionHelpers.add_quat_head_dir_degrees_column` so the existing yaw-about-Z formula computes the correct world heading for OptiTrack's default Y-up data.
todos:
  - id: edit-quat-remap
    content: In QuaternionHelpers.add_quat_head_dir_degrees_column ([NeuroPy/neuropy/core/position.py:67-91](NeuroPy/neuropy/core/position.py)), insert the Y-up to Z-up remap (x=rx, y=-rz, z=ry, w=rw) and update the docstring to record the assumed input convention. Leave the yaw formula and mod-360 normalization unchanged.
    status: completed
isProject: false
---

## Background

`posfromCSV` in [NeuroPy/neuropy/io/optitrackio.py](NeuroPy/neuropy/io/optitrackio.py) (lines 76-191) reads OptiTrack rotation columns straight through as `rx, ry, rz, rw` with no axis remap, and the active `_parse_folder` (lines 533-535) does the same for positions. OptiTrack Motive's default world frame is right-handed Y-up, so the up axis in `pos_df` is `y`/`ry`. The current `add_quat_head_dir_degrees_column` ([NeuroPy/neuropy/core/position.py:67-91](NeuroPy/neuropy/core/position.py)) computes yaw-about-Z, which is the wrong axis for that convention.

## Fix

Apply the Y-up to Z-up coordinate change (+90 deg rotation about X) to the quaternion components before evaluating the yaw formula. Algebraically this reduces to:

```text
(x, y, z, w)_zup = (rx, -rz, ry, rw)
```

This is a similarity transform `q' = p * q * p^{-1}` with `p = (sin 45, 0, 0, cos 45)`; the off-diagonal cross terms cancel and the result is the simple permutation+sign-flip above. It preserves chirality (a naive swap without the negation flips handedness and produces incorrect yaw whenever there is simultaneous pitch/roll).

## Single-file edit

In [NeuroPy/neuropy/core/position.py](NeuroPy/neuropy/core/position.py), modify only the body of `add_quat_head_dir_degrees_column` (lines 76-80). Read the four rotation columns into `rx, ry, rz, rw`, then build the local `x, y, z, w` used by the existing yaw formula as:

```python
x = rx
y = -rz
z = ry
w = rw
```

The remaining lines (the existing `siny_cosp` / `cosy_cosp` / `arctan2` / mod-360 block) stay exactly as-is. Update the docstring to record that the input quaternion is assumed to be in the OptiTrack right-handed Y-up convention and that this remap converts it to Z-up so the standard yaw-about-Z formula applies.

## Verification (no code changes; user runs)

After the edit, re-run the diagnostic from the previous turn against `pos_df`. Expected outcome:

- The `Y_up` floor-plane row (i.e. movement heading on the X/Z plane) should now correlate strongly with `quat_head_dir_degrees` (|r| in roughly the 0.6-0.95 range during running bouts).
- The `Z_up` row should correlate weakly.
- A 180 deg constant offset would indicate the rigid body's forward axis was defined as `-Z` instead of `+Z` in Motive; that is a separate calibration constant the user can subtract, not a math error in the remap.

## Out of scope

- Position columns (`x, y, z`) are not touched. The legacy `old_stuff` method in [NeuroPy/neuropy/io/optitrackio.py](NeuroPy/neuropy/io/optitrackio.py) (lines 684-686) used to swap position `y` and `z`; restoring that is a separate decision because it would affect every downstream consumer (place fields, plotting, etc.), not just heading.
- No new parameter is added (e.g. `up_axis='y'`). If multi-convention support becomes needed later, that can be added without breaking this fix.