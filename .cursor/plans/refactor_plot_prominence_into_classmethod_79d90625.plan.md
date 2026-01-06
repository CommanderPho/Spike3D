---
name: Refactor_plot_Prominence_into_classmethod
overview: Refactor the standalone plot_Prominence function into a @classmethod on PeakPromenenceDisplay and update all in-repo usages while preserving behavior.
todos:
  - id: add-classmethod
    content: Add a @classmethod plot_Prominence to PeakPromenenceDisplay in peak_prominence2d.py by moving the existing function body into the class.
    status: completed
  - id: wrap-legacy-function
    content: Refactor the standalone plot_Prominence in peak_prominence2d.py into a thin wrapper that delegates to PeakPromenenceDisplay.plot_Prominence while keeping its signature.
    status: completed
    dependencies:
      - add-classmethod
  - id: update-internal-usages
    content: Update all internal references in peak_prominence2d.py (docstring example and bottom-of-file demo) to call PeakPromenenceDisplay.plot_Prominence instead of the standalone function.
    status: completed
    dependencies:
      - add-classmethod
  - id: update-eloy-usage
    content: Update EloyAnalysis.py to import PeakPromenenceDisplay and call its plot_Prominence classmethod instead of importing/calling the standalone function.
    status: completed
    dependencies:
      - add-classmethod
  - id: verify-and-style-check
    content: Run linters/tests as appropriate to verify there are no import or runtime issues and that style (single-line function signatures where possible) is preserved.
    status: completed
    dependencies:
      - add-classmethod
      - wrap-legacy-function
      - update-internal-usages
      - update-eloy-usage
---

## Refactor `plot_Prominence` into `PeakPromenenceDisplay` classmethod

### 1. Introduce classmethod on `PeakPromenenceDisplay`

- **Add new classmethod**: In [`pyphoplacecellanalysis/External/peak_prominence2d.py`](pyphoplacecellanalysis/External/peak_prominence2d.py), inside the existing `PeakPromenenceDisplay` class, add a `@classmethod def plot_Prominence(cls, xx, yy, slab, peaks, idmap, promap, parentmap, n_contour_levels=None, debug_print=False)` whose body is the current implementation of the standalone `plot_Prominence` function (creating the 4-panel Matplotlib figure and returning `(figure, (ax1, ax2, ax3, ax4))`).
- **Preserve behavior**: Keep the plotting details identical (contour levels, lines, labels, 3D surface, prominence heatmap, `plt.show(block=False)`, and debug printing) so existing callers see no visual changes.

### 2. Turn the existing function into a thin wrapper (optional compatibility)

- **Refactor standalone `plot_Prominence`**: Replace the body of the top-level `plot_Prominence` function with a one-line call to the classmethod, e.g. `return PeakPromenenceDisplay.plot_Prominence(xx, yy, slab, peaks, idmap, promap, parentmap, n_contour_levels=n_contour_levels, debug_print=debug_print)`, keeping the signature unchanged.
- **(Optional) Deprecation note**: Optionally add a short comment or docstring note indicating that callers should migrate to `PeakPromenenceDisplay.plot_Prominence(...)` going forward, while preserving the wrapper for backward compatibility.

### 3. Update internal usages within `peak_prominence2d.py`

- **Compute wrapper usage**: In `compute_prominence_contours`’s docstring example around lines 526–533, update the sample call from `plot_Prominence(...)` to `PeakPromenenceDisplay.plot_Prominence(...)` to match the new preferred API.
- **Built-in demo block**: Near the bottom of the file (around lines 2238–2241), change the sample call `figure, (ax1, ax2, ax3, ax4) = plot_Prominence(...)` to use `PeakPromenenceDisplay.plot_Prominence(...)` instead.

### 4. Update external usage in Eloy analysis display code

- **Change import**: In [`pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/EloyAnalysis.py`](pyphoplacecellanalysis/General/Pipeline/Stages/DisplayFunctions/EloyAnalysis.py), replace `from pyphoplacecellanalysis.External.peak_prominence2d import plot_Prominence` with an import of `PeakPromenenceDisplay`.