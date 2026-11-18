---
name: Refactor pagination to use PaginationControlWidget
overview: Refactor the manual pagination widget implementation to use the existing PaginationControlWidget class, ensuring controls are created only once during initialization and properly embedded in the dock structure.
todos:
  - id: refactor_build_page_controls
    content: Refactor _build_page_controls() to use PaginationControlWidget instead of manual widget creation
    status: completed
  - id: refactor_update_visibility
    content: Update _update_page_controls_visibility() to work with PaginationControlWidget API
    status: completed
  - id: remove_slider_handler
    content: Remove _on_page_slider_changed() method (no longer needed)
    status: completed
  - id: add_jump_handler
    content: Add _on_page_jump() method to handle jump_to_page signal
    status: completed
  - id: update_page_change
    content: Simplify _on_page_change() to use PaginationControlWidget API
    status: completed
  - id: ensure_single_init
    content: Modify _update_trajectory_widget() to create pagination controls only once during initialization
    status: completed
  - id: update_type_annotation
    content: Update page_controls field type annotation to use PaginationControlWidget
    status: completed
isProject: false
---

# Refactor Pagination Controls to Use PaginationControlWidget

## Overview

Replace the manual pagination widget implementation with the existing `PaginationControlWidget` class, ensuring controls are created once during initialization and properly embedded in the dock.

## Current Issues

1. Manual pagination widget is built from scratch instead of using `PaginationControlWidget`
2. Pagination controls may be created multiple times in `_update_trajectory_widget()`
3. Controls are created conditionally based on `num_pages`, which may change

## Changes Required

### 1. Refactor `_build_page_controls()` method

**File**: `PredictiveDecodingComputations.py` (lines ~5194-5257)

Replace the manual widget creation with `PaginationControlWidget`:

- Remove manual creation of `QWidget`, `QPushButton`, `QSlider`, `QLabel`
- Create `PaginationControlWidget(n_pages=num_pages)` instead
- Connect signals: `jump_to_page`, `jump_previous_page`, `jump_next_page`
- Store only the widget reference (remove individual component storage)
- Use `programmatically_update_page_idx()` for programmatic updates
- Use `state.n_pages` and `state.current_page_idx` for state management

### 2. Refactor `_update_page_controls_visibility()` method

**File**: `PredictiveDecodingComputations.py` (lines ~5260-5278)

Update to work with `PaginationControlWidget`:

- Access widget via `page_controls['widget']` (which is now a `PaginationControlWidget`)
- Use `pagination_widget.state.n_pages` instead of separate slider/label
- Use `pagination_widget._on_update_pagination()` to update when page count changes
- Use `pagination_widget.programmatically_update_page_idx()` with `block_signals=True` for external updates

### 3. Remove `_on_page_slider_changed()` method

**File**: `PredictiveDecodingComputations.py` (lines ~5281-5293)

This method is no longer needed - `PaginationControlWidget` handles slider changes via its `jump_to_page` signal.

### 4. Update `_on_page_change()` method

**File**: `PredictiveDecodingComputations.py` (lines ~5296-5317)

Simplify to work with `PaginationControlWidget`:

- Remove manual slider value updates
- Use `pagination_widget.programmatically_update_page_idx(new_page, block_signals=True)` instead

### 5. Add `_on_page_jump()` method

**File**: `PredictiveDecodingComputations.py` (new method)

Handle the `jump_to_page` signal from `PaginationControlWidget`:

- Update `self.trajectory_active_page_idx[a_past_future_name] = page_idx`
- Call `self._refresh_trajectory_widget(a_past_future_name)`

### 6. Ensure single initialization in `_update_trajectory_widget()`

**File**: `PredictiveDecodingComputations.py` (lines ~5595-5696)

Modify the container creation logic:

- When `canvas_needs_init` is True:
  - Create pagination controls BEFORE creating container (with initial num_pages=1 as placeholder)
  - Always add pagination widget to container (even if hidden initially)
  - This ensures controls exist from the start and are only updated, not recreated
- When `canvas_needs_init` is False:
  - Only update existing controls, never create new ones
  - Remove the fallback logic that creates controls on-the-fly

### 7. Update field type annotation

**File**: `PredictiveDecodingComputations.py` (line ~4666)

Change:

```python
page_controls: Dict[str, Dict[str, Any]] = field(default=Factory(dict))
```

To:

```python
page_controls: Dict[str, Dict[str, PaginationControlWidget]] = field(default=Factory(dict))
```

## Implementation Details

### Signal Connections

```python
pagination_widget.jump_to_page.connect(lambda page_idx: self._on_page_jump(a_past_future_name, page_idx))
pagination_widget.jump_previous_page.connect(lambda: self._on_page_change(a_past_future_name, -1))
pagination_widget.jump_next_page.connect(lambda: self._on_page_change(a_past_future_name, 1))
```

### State Updates

- Use `pagination_widget.state.n_pages = num_pages` followed by `pagination_widget._on_update_pagination()` to update page count
- Use `pagination_widget.programmatically_update_page_idx(page_idx, block_signals=True)` for programmatic page changes

### Initialization Flow

1. `buildUI()` → `_build_past_widget()` / `_build_future_widget()` → creates dock
2. `update_displayed_epoch()` → `_update_trajectory_widget()` → creates container + pagination controls (first time only)
3. Subsequent updates only modify existing controls, never recreate them

## Testing Considerations

- Verify pagination controls appear only once per widget
- Verify controls are properly embedded at bottom of dock
- Verify page navigation works correctly
- Verify controls show/hide based on num_pages > 1
- Verify controls update correctly when epoch changes (page count may change)

