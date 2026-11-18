---
name: Fix pagination widget layout alignment
overview: Adjust the layout in `_update_trajectory_widget` to ensure the pagination widget is aligned to the bottom with fixed height, while the plot widget fills the remaining available space.
todos:
  - id: "1"
    content: Add setFixedHeight(21) call to pagination widget after adding it to layout
    status: completed
isProject: false
---

## Problem

The pagination widget should be aligned to the bottom of its parent container and only fill its required height (fixed height). The time sync widget (plot widget) should fill the rest of the available parent space.

## Current Implementation

In `_update_trajectory_widget` method (lines 5634-5656), the layout structure is:

- `container_layout` (QVBoxLayout) contains:
  - `plot_widget` added with `stretch=1` 
  - `control_widget` (pagination) added without stretch

## Solution

Based on the pattern used in other widgets (`PaginationMixins.py` line 681, `stacked_epoch_slices.py` lines 2033, 2411), the pagination widget should:

1. Be added to the layout (already done)
2. Have `setFixedHeight(21)` called to set a fixed height of 21 pixels
3. The plot widget already has `stretch=1` to fill remaining space

The QVBoxLayout will naturally place the pagination at the bottom since it's added after the plot widget, and with `setFixedHeight(21)`, it will only take its required height (21px) while the plot widget expands to fill the rest.

## Changes Required

### File: `PredictiveDecodingComputations.py`

**Location:** Lines 5643-5651 in `_update_trajectory_widget` method

**Changes:**

After adding the `control_widget` to the `container_layout`, add a call to set its fixed height:

```python
# Always add pagination widget to container (even if hidden initially)
if a_past_future_name in self.page_controls and 'widget' in self.page_controls[a_past_future_name]:
    control_widget = self.page_controls[a_past_future_name]['widget']
    container_layout.addWidget(control_widget)
    # Set fixed height to match pattern used in other widgets (PaginationMixins, stacked_epoch_slices)
    control_widget.setFixedHeight(21)
    # Set visibility based on actual num_pages
    control_widget.setVisible(num_pages > 1)
    # Update the widget state with actual num_pages
    if num_pages > 0:
        self._update_page_controls_visibility(a_past_future_name, num_pages)
```

This matches the pattern used in:

- `PaginationMixins._subfn_helper_add_pagination_control_widget()`: `setFixedHeight(21)`
- `stacked_epoch_slices.py` line 2033: `global_paginator_controller_widget.setFixedHeight(21)`
- `stacked_epoch_slices.py` line 2411: `a_controlled_widget.ui.thin_button_bar_widget.setFixedHeight(21)`

