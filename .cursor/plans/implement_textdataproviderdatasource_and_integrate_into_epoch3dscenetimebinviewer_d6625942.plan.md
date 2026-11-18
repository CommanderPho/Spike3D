---
name: Implement TextDataProviderDatasource and integrate into Epoch3DSceneTimeBinViewer
overview: Fix the incomplete TextDataProviderDatasource class to properly filter by epoch_idx and t_bin_idx and return formatted text, then integrate it as an optional property of Epoch3DSceneTimeBinViewer that takes precedence over the existing text_columns approach.
todos:
  - id: fix_imports
    content: Add missing import for attrs (define, field) at top of file
    status: completed
  - id: fix_textdataprovider
    content: Fix TextDataProviderDatasource.get_text_label to filter by epoch_idx and t_bin_idx and return formatted text string
    status: completed
    dependencies:
      - fix_imports
  - id: add_parameter
    content: Add optional text_data_provider parameter to Epoch3DSceneTimeBinViewer.__init__
    status: completed
  - id: modify_get_text_label
    content: Modify _get_text_label_string to use text_data_provider if provided, otherwise fall back to text_columns
    status: completed
    dependencies:
      - add_parameter
      - fix_textdataprovider
---

# Implement TextDataProviderDatasource and integrate into Epoch3DSceneTimeBinViewer

## Overview

The `TextDataProviderDatasource` class is currently incomplete - it needs to properly filter a dataframe by `epoch_idx` and `t_bin_idx` and return formatted text. This will then be integrated into `Epoch3DSceneTimeBinViewer` as an optional alternative to the existing `text_columns` approach.

## Implementation Steps

### 1. Fix TextDataProviderDatasource class

**File**: [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py)

- Add missing import: `from attrs import define, field` at the top of the file
- Fix `get_text_label` method (lines 27-29):
- Filter dataframe by both `epoch_idx` and `t_bin_idx` 
- Extract the matching row
- Format all columns except `epoch_idx` and `t_bin_idx` into a text string
- Return the formatted string, or `None` if no match found
- Format should be similar to existing `_get_text_label_string`: `"col1: value1, col2: value2"` for numeric values, handle NaN values as "N/A"
- Keep `on_update_epoch_idx` as-is (it's already correct)

### 2. Integrate TextDataProviderDatasource into Epoch3DSceneTimeBinViewer

**File**: [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py)

- Add optional parameter to `__init__` (around line 684):
- `text_data_provider: Optional[TextDataProviderDatasource] = None`
- Store as instance variable: `self.text_data_provider = text_data_provider`
- Modify `_get_text_label_string` method (around line 1494):
- Check if `self.text_data_provider` is provided
- If provided, call `text_data_provider.get_text_label(self.curr_epoch_idx, t_bin_idx)` and return the result
- Otherwise, fall back to existing `text_columns` logic (current implementation)
- No changes needed to label rendering code - it already calls `_get_text_label_string`

## Data Structure Assumptions

- The dataframe in `TextDataProviderDatasource` should have:
- `epoch_idx`: integer column for epoch index
- `t_bin_idx`: integer column for time bin index within epoch
- Additional data columns that will be formatted into text labels
- Formatting: All columns except `epoch_idx` and `t_bin_idx` will be formatted as `"col_name: value"` pairs, with appropriate formatting for numeric/NaN values

## Testing Considerations

- Test with dataframe containing `epoch_idx`, `t_bin_idx`, and multiple data columns