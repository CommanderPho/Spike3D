---
name: Add time bin posterior heatmaps row
overview: Add a row of tiny heatmaps below the main posterior heatmap in the central dock, showing individual time bin posteriors (up to 10 at a time) for the active epoch.
todos:
  - id: modify_figure_layout
    content: Change figure layout from single subplot to GridSpec with main heatmap on top and row of tiny heatmaps below
    status: completed
  - id: extract_time_bin_data
    content: Extract individual time bin posteriors from p_x_given_n, limiting to first 10 time bins
    status: completed
  - id: create_tiny_heatmap_subplots
    content: Create horizontal row of tiny subplots for time bin heatmaps with minimal labels
    status: completed
  - id: plot_time_bin_heatmaps
    content: Plot each time bin posterior as a tiny heatmap with shared color scale
    status: completed
  - id: update_both_branches
    content: Update both initialization (needed_init) and update branches to use new layout
    status: completed
---

## Plan: Add Time Bin Posterior Heatmaps Row to PredictiveDecodingDisplayWidget

### Overview

Modify the `decoded_posterior` dock in `PredictiveDecodingDisplayWidget` to display a row of tiny heatmaps below the main summed posterior heatmap. Each tiny heatmap shows the posterior for a single time bin within the active epoch.

### Current State

- The `decoded_posterior` dock currently shows a single heatmap that sums all time bins: `posterior_2d = np.sum(p_x_given_n, axis=2)`
- The posterior data `p_x_given_n` has shape `(n_x_bins, n_y_bins, n_time_bins)` for each epoch
- The figure is created with a single subplot (`fig.add_subplot(111)`)
- Located in `update_displayed_epoch()` method around lines 2971-3060

### Implementation Steps

1. **Modify figure layout** (lines ~2994-2995, ~3050-3051):

- Change from single subplot to a `GridSpec` layout with:
- Top subplot: Main summed posterior heatmap (larger, takes most space)
- Bottom subplot: Row of tiny heatmaps (small, horizontal row)
- Use `matplotlib.gridspec.GridSpec` to create a 2-row layout where the top row is much larger

2. **Extract time bin posteriors** (around line 2983):

- When `p_x_given_n` is available, extract individual time bin slices: `p_x_given_n[:, :, t_bin_idx]` for each time bin
- Limit to first 10 time bins: `num_time_bins_to_show = min(10, p_x_given_n.shape[2])`
- Handle the case when `epoch_high_prob_pos_masks` is used (may need to skip time bin display or use alternative data)

3. **Create tiny heatmap subplots** (in both `needed_init` and update branches):

- Create a horizontal row of subplots using `GridSpec` for the bottom section
- Each subplot should be very small (e.g., 1/10th the width of the main plot)
- Remove axes labels/ticks from tiny heatmaps to save space
- Add minimal labels (e.g., "t=0", "t=1", etc.) below each tiny heatmap

4. **Plot individual time bin heatmaps**:

- For each of the first 10 time bins, plot `p_x_given_n[:, :, t_bin_idx]` as a tiny heatmap
- Use the same `extent`, `cmap='viridis'`, and `origin='lower'` as the main heatmap
- Share the same color scale (vmin/vmax) across all tiny heatmaps for consistency
- Optionally: use a shared colorbar or individual colorbars per tiny heatmap (shared is cleaner)

5. **Update both initialization and update paths**:

- Modify the `needed_init` branch (lines ~2991-3044) to create the new layout
- Modify the update branch (lines ~3045-3060) to update both the main heatmap and time bin heatmaps
- Ensure axes are properly cleared and recreated on updates

### Key Files to Modify

- [`PredictiveDecodingComputations.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\PredictiveDecodingComputations.py) - Lines ~2971-3060 in `update_displayed_epoch()` method

### Design Considerations

- **Size**: Tiny heatmaps should be small enough that 10 fit horizontally without crowding
- **Layout**: Use `GridSpec` with height ratios like `[3, 1]` or `[4, 1]` to give most space to main heatmap
- **Color scale**: Share vmin/vmax across all tiny heatmaps for visual consistency
- **Performance**: Only extract and plot the first 10 time bins to keep rendering fast
- **Edge cases**: Handle epochs with fewer than 10 time bins gracefully
- **Code locality**: Keep all changes within the `decoded_posterior` section of `update_displayed_epoch()` method

### Future Enhancements (Not in this iteration)

- Add scrolling/navigation to view different sets of 10 time bins for longer epochs
- Add click interaction to select a time bin and update the main view
- Add time bin labels with actual time values instead of indices