---
name: Position filtering sequence visualizer
overview: Create a high-performance visualizer function that displays temporal sequences of position dataframes at each filtering stage, showing contiguous epochs and gaps between them. The visualizer will heavily downsample points for performance and use matplotlib for rendering.
todos:
  - id: create_downsample_helper
    content: Create helper function to downsample position dataframes while preserving epoch boundaries and time structure
    status: completed
  - id: create_epoch_detection_helper
    content: Create helper function to detect contiguous epochs and gaps from time points in each filtering stage
    status: completed
  - id: implement_visualize_function
    content: Implement main visualize_filtering_stages function with subplot layout, downsampling, epoch highlighting, and statistics
    status: completed
    dependencies:
      - create_downsample_helper
      - create_epoch_detection_helper
  - id: add_integration_point
    content: Add optional call to visualizer in detect_matching_past_future_positions method (commented out by default for debugging)
    status: completed
    dependencies:
      - implement_visualize_function
---

# Position Filtering Sequence Visualizer

## Overview

Create a high-performance visualizer to debug position filtering steps by showing temporal sequences, contiguous epochs, and gaps at each filtering stage.

## Implementation Details

### Location

Add the visualization function to [`PredictiveDecodingComputations.py`](pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/PredictiveDecodingComputations.py) in the `MatchingPastFuturePositionsResult` class (around line 1022) or as a standalone classmethod/function.

### Function Signature

```python
@classmethod
def visualize_filtering_stages(cls, measured_positions_df: pd.DataFrame, relevant_positions_df_after_merge: pd.DataFrame, relevant_positions_df_final: pd.DataFrame, epoch_high_prob_mask: NDArray, curr_epoch_start_t: float, curr_epoch_stop_t: float, max_points_per_plot: int = 5000, figsize: Tuple[int, int] = (16, 10), ax: Optional[plt.Axes] = None) -> plt.Figure:
```

### Key Features

1. **Temporal Sequence Visualization**

   - X-axis: time (`t` column)
   - Y-axis: Different rows for each filtering stage
   - Show included/excluded points at each stage

2. **Filtering Stages to Visualize**

   - Stage 0: `measured_positions_df` (initial, all positions)
   - Stage 1: After merge with `an_epoch_mask_included_binned_x_y_columns_idx_df` (spatial filter)
   - Stage 2: After `pos_matches_epoch_mask` filter (final spatial filter)
   - Optional: Past/present/future categorization

3. **Performance Optimizations**

   - Heavy downsampling using `max_points_per_plot` parameter (default 5000)
   - Use numpy array operations instead of pandas iterrows
   - Efficient epoch detection using existing `compute_matching_pos_epochs_df` method
   - Use matplotlib's efficient plotting (scatter with alpha for density)

4. **Epoch and Gap Visualization**

   - Detect contiguous epochs using `compute_matching_pos_epochs_df` or similar
   - Highlight epochs as colored rectangles/regions
   - Show gaps between epochs as white/transparent regions
   - Use different colors for past/present/future if applicable

5. **Visual Design**

   - Subplot layout: One row per filtering stage
   - Time axis synchronized across all subplots
   - Color-coded: included points (blue/green), excluded points (red/gray), epochs (highlighted regions)
   - Add vertical lines for `curr_epoch_start_t` and `curr_epoch_stop_t`
   - Include statistics text (count of points, epochs, gaps)

### Implementation Steps

1. **Downsampling Function**

   - Create helper to downsample time series while preserving epoch boundaries
   - Use uniform sampling or adaptive sampling (more points in dense regions)

2. **Epoch Detection**

   - For each stage, detect contiguous epochs from time points
   - Use gap detection (time differences > threshold) to identify epoch boundaries

3. **Plotting**

   - Create subplots for each filtering stage
   - Plot downsampled time points as scatter or line plots
   - Overlay epoch rectangles using `axvspan` or `Rectangle` patches
   - Add labels, legends, and statistics

4. **Integration**

   - Make it easy to call during debugging
   - Return matplotlib figure for further customization
   - Support both standalone plotting and embedding in existing figures

### Usage Example

```python
# In detect_matching_past_future_positions method, after filtering steps:
fig = MatchingPastFuturePositionsResult.visualize_filtering_stages(
    measured_positions_df=measured_positions_df,
    relevant_positions_df_after_merge=relevant_positions_df_after_merge,
    relevant_positions_df_final=relevant_positions_df,
    epoch_high_prob_mask=epoch_high_prob_mask,
    curr_epoch_start_t=curr_epoch_start_t,
    curr_epoch_stop_t=curr_epoch_stop_t,
    max_points_per_plot=5000
)
plt.show()
```

### Dependencies

- matplotlib (already imported)
- numpy (already imported)
- pandas (already imported)
- Existing epoch detection utilities from the codebase