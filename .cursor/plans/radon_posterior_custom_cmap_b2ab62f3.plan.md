---
name: Radon posterior custom cmap
overview: Wire `RadonTransformDebugger` posterior images to accept `posterior_heatmap_imshow_kwargs` (matplotlib-style `cmap`/`vmin`/`vmax`), defaulting to the greyscale low-values-dropped colormap, by converting matplotlib colormaps into silx `Colormap` LUTs.
todos:
  - id: add-field-helper
    content: Add posterior_heatmap_imshow_kwargs field (greyscale default) and matplotlib→silx Colormap helper
    status: completed
  - id: wire-addImage
    content: Use kwargs in perform_add_real_space_posterior / add_real_space_posterior when building the silx Colormap for addImage
    status: completed
isProject: false
---

# Enable custom posterior cmap in RadonTransformDebugger

## Context

[`perform_add_real_space_posterior`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py) currently hardcodes:

```python
a_cmap = Colormap(name="viridis", vmin=0)
```

Elsewhere (e.g. [`DecoderPredictionError.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\DisplayFunctions\DecoderPredictionError.py)), posteriors use:

```python
active_cmap = FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.01, full_opacity_threshold=0.25)
posterior_heatmap_imshow_kwargs=dict(cmap=active_cmap)
```

Silx cannot take a matplotlib `LinearSegmentedColormap` directly; it accepts `Colormap(colors=Nx4_LUT, vmin=..., vmax=...)` (RGBA float/`uint8` LUT). Alpha in the greyscale cmap will be preserved via the Nx4 LUT.

## Approach (chosen default)

Default the debugger to the greyscale dropped-low-values cmap so existing `build_GUI()` calls pick it up with no notebook edits. Keep `posterior_heatmap_imshow_kwargs` overridable so callers can still pass a different `cmap` / `vmin` / `vmax`.

## Changes — only [`RadonTransformDebuggerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py)

1. **Import** `FixedCustomColormaps` from `LongShortDisplayConfig`.

2. **Add field** on `RadonTransformDebugger`:

```python
posterior_heatmap_imshow_kwargs: Dict = field(default=Factory(lambda: dict(
    cmap=FixedCustomColormaps.get_custom_greyscale_with_low_values_dropped_cmap(low_value_cutoff=0.01, full_opacity_threshold=0.25),
)))
```

3. **Add helper** `matplotlib_cmap_to_silx_colormap(mpl_cmap, vmin=0, vmax=None, n_colors=256) -> Colormap`:
   - If `mpl_cmap` is already a silx `Colormap`, return it (apply vmin/vmax if provided).
   - If it is a string name known to silx, use `Colormap(name=..., vmin=..., vmax=...)`.
   - Otherwise sample `mpl_cmap(np.linspace(0, 1, n_colors))` to an Nx4 LUT and build `Colormap(colors=lut, vmin=vmin, vmax=vmax)`.

4. **Thread kwargs through plotting**:
   - Extend `perform_add_real_space_posterior(..., posterior_heatmap_imshow_kwargs=None)`.
   - Resolve `cmap` / `vmin` / `vmax` from that dict (defaults: greyscale cmap, `vmin=0`, `vmax=None`).
   - Build silx colormap via the helper; pass it to `addImage(..., colormap=a_cmap, ...)`.
   - `add_real_space_posterior` forwards `self.posterior_heatmap_imshow_kwargs`.

5. **No notebook edits** — default field covers the open `SCRATCH/2024-04-20 - silx_RadonTransformTesting.ipynb` usage. Callers can still do:

```python
dbgr.posterior_heatmap_imshow_kwargs = dict(cmap=active_cmap)
# or at construction:
RadonTransformDebugger(..., posterior_heatmap_imshow_kwargs=dict(cmap=active_cmap))
```

Overlay colors (orange band, red line, white rho/score markers) stay as-is; they remain readable on the greyscale posterior.
