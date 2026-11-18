---
name: Fix export width scaling
overview: "Fix Radon PDF export sizing to match PosteriorExporting image exports: height defaults to n_pos (1 px per xbin), width = height * n_t / n_pos with no min-width clamp, and pass dpi into saveGraph."
todos:
  - id: fix-figsize-formula
    content: "Match PosteriorExporting: height=n_pos when None; width=height*n_t/n_pos; drop min-width clamp"
    status: completed
  - id: pass-dpi-savegraph
    content: Pass export_dpi into saveGraph; update field defaults
    status: completed
isProject: false
---

# Fix Radon export width to track time bins

## Root cause

Current sizing in [`_export_figsize_inches`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py):

```python
width_px = max(export_min_width_px, int(export_desired_height_px * n_t / max(n_pos, 1)))
# defaults: height=400, min_width=200, n_pos≈60
# → width ≈ 6.67*n_t, but clamped to 200 for all n_t ≲ 30
```

Typical ripple bin counts all hit the clamp → every PDF looks the same width.

## Correct reference ([`PosteriorExporting`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\data_exporting.py) + [`get_array_as_image`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\plotting\media_output_helpers.py))

```python
if desired_height is None:
    desired_height = n_xbins  # 1 pixel per position bin
desired_width = int(desired_height * n_time_bins / n_pos_bins)  # → equals n_t when height == n_pos
```

No minimum-width clamp. Longer epochs get proportionally wider images.

## Fix — only [`RadonTransformDebuggerWidget.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Silx\RadonTransformDebuggerWidget.py)

1. **Fields**
   - Change `export_desired_height_px` to `Optional[int] = None` (None → use `n_pos`, same as PosteriorExporting).
   - Remove `export_min_width_px` (it breaks proportionality).
   - Keep `export_dpi` (default `100.0`).

2. **`_export_figsize_inches`**
   ```python
   n_pos, n_t = p.shape[0], p.shape[1]
   height_px = int(self.export_desired_height_px) if self.export_desired_height_px is not None else n_pos
   width_px = int(height_px * n_t / max(n_pos, 1))  # no clamp
   return (width_px / dpi, height_px / dpi)
   ```

3. **`_save_publication_pdf`**
   - Pass dpi into silx: `a_plot.saveGraph(..., fileFormat='pdf', dpi=self.export_dpi)` so the pixel/inch mapping matches the figsize we set (silx forwards `dpi` to `fig.savefig`).

For taller labeled publication figures, callers set e.g. `dbgr.export_desired_height_px = 400` or `1200`; width still scales as `height * n_t / n_pos`.
