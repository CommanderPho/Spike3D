---
name: Cmap taper toggle
overview: Add an attrs field `enable_cmap_time_distance_taper` on `PredictiveDecodingVispyWidget` (default True for backward compatibility) and branch the colorbar rectangle opacity so disabling it uses full alpha (1.0).
todos:
  - id: add-field
    content: Add enable_cmap_time_distance_taper = field(default=True) to PredictiveDecodingVispyWidget
    status: completed
  - id: branch-colorbar
    content: Wrap colorbar opacity in if self.enable_cmap_time_distance_taper else 1.0
    status: completed
  - id: optional-init-datasource
    content: (Optional) Pass **kwargs through init_from_datasource cls() for parity with init_from_list
    status: completed
isProject: false
---

# Colorbar time-distance taper toggle

## Context

The colorbar loop in `[predicitive_decoding_vispy.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` sets per-segment opacity with:

```python
opacity = np.clip(1.0 - distance_normalized * 0.8, 0.2, 1.0)
```

This should run only when the new flag is enabled.

## Changes

1. **Declare the field** on `[PredictiveDecodingVispyWidget](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` next to other display toggles (e.g. after `show_full_position_background` ~line 112):
  - `enable_cmap_time_distance_taper: bool = field(default=True)`
   Default **True** keeps today’s behavior unchanged.
2. **Branch in the colorbar loop** (~lines 1263–1269): if `self.enable_cmap_time_distance_taper`, keep the existing `np.clip(...)` line; else set `opacity = 1.0`. Keep computing `distance_normalized` only where needed, or always compute it and ignore when taper is off (simplest: one `if/else` on `opacity` only).
3. **Optional UX** (only if you want labels to stay accurate): the line ~1283 `title_opacity` (“Opacity: 1.0 (close) → 0.2 (distant)”) could be omitted or replaced when `not self.enable_cmap_time_distance_taper`. Not required for the feature; can be a tiny follow-up.

## Factory / call sites

- `[init_from_list](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` already passes `**kwargs` into `cls(...)`, so callers can use `enable_cmap_time_distance_taper=False` without changing its signature.
- `[init_from_datasource](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\predicitive_decoding_vispy.py)` does not forward `**kwargs`; add `**kwargs` to that `cls(...)` call if you want the same override there; otherwise the default applies.

## Out of scope (per your snippet)

Trajectory opacity in `_render_trajectory_side` (~903) is unchanged unless you ask to tie it to the same flag later.