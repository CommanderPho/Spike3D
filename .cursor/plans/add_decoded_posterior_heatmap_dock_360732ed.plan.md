---
name: Add decoded posterior heatmap dock
overview: Add a third dock centered between the existing 'past' and 'future' docks in PredictiveDecodingDisplayWidget that displays the decoded posterior heatmap for the selected epoch.
todos:
  - id: modify_init_ui
    content: Update init_UI method to add 'decoded_posterior' dock to the category list between 'past' and 'future'
    status: completed
  - id: add_heatmap_plotting
    content: Add heatmap plotting code in update_displayed_epoch to display p_x_given_n_list[an_epoch_idx] as a 2D heatmap (sum over time dimension)
    status: completed
    dependencies:
      - modify_init_ui
  - id: embed_heatmap_widget
    content: Embed the matplotlib heatmap figure in the decoded_posterior dock widget following the same pattern as past/future docks
    status: completed
    dependencies:
      - add_heatmap_plotting
---

# Add Decoded Posterior Heatmap Dock to PredictiveDecodingDisplayWidget

## Overview

Modify `PredictiveDecodingDisplayWidget` to add a third dock that displays `self.decoded_result.p_x_given_n_list[an_epoch_idx]` as a heatmap, positioned between the existing 'past' and 'future' trajectory docks.

## Changes Required

### 1. Modify `init_UI` method (lines 2276-2311)

- Add a third dock category 'decoded_posterior' to the dock creation sequence
- Update the loop to create three docks: 'past', 'decoded_posterior', 'future'
- Position 'decoded_posterior' between 'past' and 'future' using dock positioning

**Specific changes:**

- Change the loop from `for category_name in ['past', 'future']:` to `for category_name in ['past', 'decoded_posterior', 'future']:`
- The dock positioning logic (lines 2301-2304) will automatically handle the three-dock layout

### 2. Modify `update_displayed_epoch` method (lines 2314-2417)

- Add a special case to handle 'decoded_posterior' category
- Extract `p_x_given_n = self.decoded_result.p_x_given_n_list[an_epoch_idx] `which has shape `(n_x_bins, n_y_bins, n_time_bins)`
- Create a matplotlib figure with a heatmap using `imshow`
- For the 3D array, sum or average over the time dimension to create a 2D heatmap, or show a specific time bin (e.g., middle time bin)
- Use `xbin_centers` and `ybin_centers` for the extent parameter
- Embed the figure in the dock widget similar to how past/future plots are embedded (lines 2377-2410)

**Specific implementation:**

- Add code after the existing loop (around line 2412) or modify the loop to handle 'decoded_posterior' separately
- Use matplotlib's `imshow` with extent based on `self.xbin_centers` and `self.ybin_centers`
- Sum over time axis: `posterior_2d = np.sum(p_x_given_n, axis=2)` or use middle time bin: `posterior_2d = p_x_given_n[:, :, p_x_given_n.shape[2] // 2]`
- Add colorbar for the heatmap
- Follow the same widget embedding pattern as past/future docks

## Files to Modify

- `pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py`

## Implementation Notes

- The decoded posterior data `p_x_given_n_list[an_epoch_idx] `is a 3D array `(n_x_bins, n_y_bins, n_time_bins)`
- For display, we'll sum over the time dimension to create a 2D heatmap showing the total probability distribution across the epoch