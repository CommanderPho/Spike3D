---
name: RainbowHue colormap restore
overview: Add a first-class `rainbow_hue` predictive-time colormap (HSV hue cycle with the same s/v as the current hardcoded loop), restore `predictive_time_rgb` and `predictive_time_bin_rgba` to dispatch through `_predictive_time_rgb_for_key`, and drop the temporary `s`/`v` parameters from the public API.
todos:
  - id: register-rainbow-hue
    content: Add rainbow_hue to Literal, names tuple, mpl map, and HSV s/v constants
    status: completed
  - id: route-rgb-for-key
    content: Early-return rainbow_hue in _predictive_time_rgb_for_key before cyan_magenta mpl None branch
    status: completed
  - id: restore-public-api
    content: Fix predictive_time_rgb and predictive_time_bin_rgba; trim docstrings
    status: completed
isProject: false
---

# RainbowHue colormap and general predictive-time sampling

## Context

[predictive_time_colormap.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predictive_time_colormap.py) already contains the correct plumbing (`_predictive_time_rgb_for_key`, `_resolve_colormap_key`, matplotlib vs. custom cyan–magenta stops). `predictive_time_rgb` and `predictive_time_bin_rgba` were temporarily overridden to always use full-spectrum HSV (`colorsys.hsv_to_rgb`), ignoring `colormap`.

Call sites ([predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py), [predictive_decoding_central_view.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predictive_decoding_central_view.py), [vispy_raster.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_raster.py)) only pass `u` / `alpha` and optional `colormap`; none use `s`/`v`, so removing those kwargs is safe.

## Implementation (single file)

1. **Register `rainbow_hue`**
  - Extend `PredictiveTimeColormapName` with `"rainbow_hue"`.
  - Add `"rainbow_hue"` to `PREDICTIVE_TIME_COLORMAP_NAMES` and to `_PREDICTIVE_TIME_MPL_NAME` with value `None` (custom, not a matplotlib named cmap).
  - Add two module-level constants for the legacy look, e.g. `_RAINBOW_HUE_S = 0.8`, `_RAINBOW_HUE_V = 0.9` (matching the current hardcoded loop).
2. **Route rainbow in `_predictive_time_rgb_for_key`**
  - After clamping `uu` to `[0, 1]`, if `key == "rainbow_hue"`, `return colorsys.hsv_to_rgb(uu, _RAINBOW_HUE_S, _RAINBOW_HUE_V)`.
  - Keep existing branches: `mpl is None` and `key != "rainbow_hue"` still means cyan–magenta piecewise stops; non-`None` `mpl` still uses `matplotlib.colormaps[mpl](uu)`.
3. **Restore public helpers**
  - `**predictive_time_rgb`**: signature `def predictive_time_rgb(u: float, colormap: PredictiveTimeColormapName | str | None = None) -> tuple[float, float, float]:` — body `return _predictive_time_rgb_for_key(_resolve_colormap_key(colormap), float(u))` (or keep the existing finite/clip pattern inside `_predictive_time_rgb_for_key` only; no duplicate clamp in `predictive_time_rgb` if `_predictive_time_rgb_for_key` already handles non-finite `u` and clipping).
  - Fix docstring: remove the stale HSV one-liner; describe sampling `u` with the active or passed `colormap`.
  - `**predictive_time_bin_rgba`**: uncomment and keep the loop that uses `_predictive_time_rgb_for_key` with `u = t_idx / denom`, `denom = max(n, 1)`; remove the duplicate second docstring and the HSV-only loop (lines 118–124).
  - Preserve behavior for `n_bins <= 0`: return shape `(0, 4)` as in the commented block.
4. **Docstring at top of module**
  - Mention `rainbow_hue` alongside cyan–magenta and matplotlib options.
5. **Default colormap**
  - Leave `_active_predictive_time_colormap = "cool"` unless you explicitly want rainbow as default (no change keeps current behavior).

## Optional user workflow

To get the old rainbow look globally: `set_predictive_time_colormap("rainbow_hue")`. Per-call: `predictive_time_bin_rgba(n, colormap="rainbow_hue")`.

No other files require edits.