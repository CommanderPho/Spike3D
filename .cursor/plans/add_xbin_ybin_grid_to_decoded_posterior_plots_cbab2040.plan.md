---
name: Add xbin ybin grid to decoded posterior plots
overview: Add custom xbin/ybin grid lines to the decoded posterior plots (main axis and all tiny time bin axes) using the existing `_helper_add_bin_grid_lines` helper function from `DecodedTrajectoryMatplotlibPlotter`.
todos:
  - id: "1"
    content: Replace manual grid setup in _subfn_plot_posterior_with_potential_overlay with _helper_add_bin_grid_lines call
    status: completed
  - id: "2"
    content: Ensure xbin, ybin, xbin_centers, ybin_centers are accessible in the nested function scope
    status: completed
  - id: "3"
    content: Test that grid appears on both main axis and tiny time bin axes
    status: completed
isProject: false
---

## Plan: Add xbin/ybin Grid to Decoded Posterior Plots

### Overview

The `_update_posterior_widget` method currently uses a manual grid approach (lines 5705-5733) that sets ticks and uses matplotlib's built-in grid. We need to replace this with the existing `_helper_add_bin_grid_lines` helper function that's already used in the trajectory widgets (line 5577).

### Changes Required

1. **In `_subfn_plot_posterior_with_potential_overlay` function (lines 5696-5793)**:
  - Remove or replace the manual grid setup code (lines 5705-5733) that currently:
    - Sets `ax.set_axisbelow('line')`
    - Sets `ax.set_xticks(xbin)` and `ax.set_yticks(ybin)`
    - Calls `ax.grid()` with major grid
    - Configures tick parameters
  - Add a call to `DecodedTrajectoryMatplotlibPlotter._helper_add_bin_grid_lines()` after the heatmap plotting (after line 5762), passing:
    - `an_ax=ax`
    - `xbin=xbin` (from outer scope)
    - `ybin=ybin` (from outer scope)
    - `xbin_centers=xbin_centers` (from outer scope)
    - `ybin_centers=ybin_centers` (from outer scope)
    - `rotate_to_vertical=posterior_should_use_flipped` (from outer scope)
2. **Ensure grid is added to main axis**:
  - The call in `_subfn_plot_posterior_with_potential_overlay` will automatically apply to `ax_main` when called at line 5863
3. **Ensure grid is added to tiny axes**:
  - The call in `_subfn_plot_posterior_with_potential_overlay` will automatically apply to each `ax_tiny` when called in the loop at line 5880

### Implementation Details

- The `_helper_add_bin_grid_lines` method is a classmethod, so it can be called directly on `DecodedTrajectoryMatplotlibPlotter` class (already imported at line 4672)
- The helper function uses minor ticks for grid lines, which is more efficient than the current major tick approach
- The helper function already handles hiding tick marks and labels while keeping the grid visible
- The function signature: `_helper_add_bin_grid_lines(cls, an_ax, xbin=None, ybin=None, xbin_centers=None, ybin_centers=None, rotate_to_vertical: bool=False, grid_kwargs: Optional[Dict[str, Any]]=None)`

### Files to Modify

- `[PredictiveDecodingComputations.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py)`:
  - Modify `_subfn_plot_posterior_with_potential_overlay` function (lines 5696-5793)
  - Replace manual grid code with call to `_helper_add_bin_grid_lines`

