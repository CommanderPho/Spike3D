---
name: Add Locality Measures Table Tab
overview: Add a new tab to the SceneWindow sidebar that displays `locality_measures_df` as a table when in point-like mode (when the dataframe contains a 't' column). The table will highlight and scroll to the row matching the currently selected time bin.
todos:
  - id: detect_point_mode
    content: Add flag to detect point-like mode (when 't' column exists in locality_measures_df)
    status: completed
  - id: access_sidebar
    content: Access the SceneWindow sidebar widget (QTabWidget) to add new tab
    status: completed
    dependencies:
      - detect_point_mode
  - id: create_table_widget
    content: Create QTableWidget and populate with locality_measures_df data
    status: completed
    dependencies:
      - detect_point_mode
  - id: add_table_tab
    content: Add table widget as new tab to sidebar (e.g., 'Locality Measures')
    status: completed
    dependencies:
      - access_sidebar
      - create_table_widget
  - id: implement_highlighting
    content: Implement row highlighting and scrolling when epoch/time bin changes
    status: completed
    dependencies:
      - add_table_tab
  - id: integrate_highlighting
    content: Call highlighting method from on_epoch_changed when in point-like mode
    status: completed
    dependencies:
      - implement_highlighting
---

# Add Locality Measures Table Tab to SceneWin

dow Sidebar

## Overview

Add a new tab to the right sidebar of `Epoch3DSceneTimeBinViewer` that displays `locality_measures_df` as a table when in point-like mode (when the dataframe contains a 't' column). The table will highlight and scroll to the row corresponding to the currently selected time bin.

## Implementation Details

### 1. Detect Point-like Mode

- Check if `locality_measures_df` exists and contains a 't' column (point-like mode)
- This check should be done in `__init__` and stored as a flag

### 2. Access SceneWindow Sidebar

- The `SceneWindow` from silx has a built-in sidebar with tabs ("Object parameters" and "Global parameters")
- Access the sidebar widget (likely a `QTabWidget`) through SceneWindow's internal structure
- Methods to try:
- `self.scene_window.getSidebar()` (if available)
- Traverse children to find `QTabWidget`
- Access through `self.scene_window.centralWidget()` or similar

### 3. Create Table Widget

- Create a `QTableWidget` to display `locality_measures_df`
- Populate with all columns and rows from the dataframe
- Set appropriate column headers
- Make it read-only for data display
- Enable row selection and scrolling

### 4. Add Tab to Sidebar

- Add the table widget as a new tab (e.g., "Locality Measures" or "Data Table")
- Position it next to "Object parameters" tab
- Only create/add the tab when in point-like mode

### 5. Implement Row Highlighting

- When epoch changes (`on_epoch_changed`), find the matching row in the table
- Match based on the time bin center time and the 't' column in the dataframe
- Highlight the matching row (set background color)
- Scroll to make the highlighted row visible
- Use the same matching logic as `_match_time_bin_to_dataframe_row` (lines 768-782)

### 6. Handle Edge Cases

- Only show the tab when `locality_measures_df` is not None and has 't' column
- Handle cases where no matching row is found (don't highlight anything)
- Ensure the table updates properly when epoch changes

## Files to Modify

- [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py)
- Add table widget creation in `__init__` (around line 687-691)
- Add method to access sidebar and create table tab
- Add method to highlight matching row in table
- Update `on_epoch_changed` to trigger row highlighting (around line 993-1007)

## Key Code Locations