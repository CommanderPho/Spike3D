---
name: Add Custom Logging Dock Widget to Napari
overview: Add a custom scrolling text/console dock widget to the Napari window for custom logging. The widget will be integrated with the existing `_log_to_console()` method and will fail gracefully if it cannot be created.
todos:
  - id: create_logging_widget
    content: Create CustomLoggingWidget class (or adapt existing LogViewer) with append_log() method for scrolling text display
    status: pending
  - id: add_dock_method
    content: Add add_custom_logging_dock() method to DecodedTrajectoryNapariPlotter that creates and adds the dock widget with graceful error handling
    status: pending
    dependencies:
      - create_logging_widget
  - id: add_widget_field
    content: Add custom_logging_widget field to DecodedTrajectoryNapariPlotter class definition
    status: pending
  - id: integrate_log_method
    content: Modify _log_to_console() method to also write to custom_logging_widget if available, with graceful error handling
    status: pending
    dependencies:
      - add_widget_field
      - add_dock_method
  - id: optional_auto_create
    content: Add optional create_logging_dock parameter to build_ui() method for automatic dock widget creation
    status: pending
    dependencies:
      - add_dock_method
---

# Add Custom Logging Dock Widget to Napari Window

## Overview

Add a custom dock widget containing a scrolling text/console widget to the Napari window for displaying custom log messages. This will integrate with the existing `_log_to_console()` method to provide a dedicated logging area in the Napari interface.

## Implementation Details

### 1. Create Custom Logging Widget Class

Create a new class `CustomLoggingWidget` (or use existing `LogViewer` from `pyphoplacecellanalysis.GUI.Qt.Widgets.LogViewerTextEdit`) that:

- Extends `QWidget` or uses `QTextEdit`/`QPlainTextEdit` for scrolling text display
- Provides `append_log(message: str)` method to add messages
- Auto-scrolls to bottom when new messages are added
- Optionally includes a "Clear" button
- Is read-only (users can't edit, only view)

**Location**: Can be defined as a nested class in `DecodedTrajectoryNapariPlotter` or as a separate helper class.

### 2. Add Dock Widget to DecodedTrajectoryNapariPlotter

Add a new field and method to `DecodedTrajectoryNapariPlotter`:

- Field: `custom_logging_widget: Optional[Any] = field(default=None)` to store reference to the widget
- Method: `add_custom_logging_dock(self, area: str = 'bottom', name: str = 'Custom Log') -> Optional[Any]` that:
- Creates the custom logging widget
- Adds it as a dock widget to `viewer.window` using `add_dock_widget()`
- Stores reference in `self.custom_logging_widget`
- Returns the widget if successful, `None` if it fails
- **Fails gracefully**: Wraps all operations in try-except, logs errors but doesn't raise exceptions

### 3. Integrate with `_log_to_console()` Method

Modify the existing `_log_to_console()` method to also write to the custom logging widget:

- After writing to stdout and Napari console
- Check if `self.custom_logging_widget` exists
- If it exists and has an `append_log()` method, call it
- Wrap in try-except to fail gracefully if widget is unavailable or method doesn't exist

### 4. Optional Auto-Creation

Add an optional parameter to `build_ui()` method:

- `create_logging_dock: bool = False` - if True, automatically creates the logging dock widget
- This allows users to opt-in to having the dock widget created automatically

## Files to Modify

- `pyphoplacecellanalysis/PhoPositionalData/plotting/mixins/decoder_plotting_mixins.py` (DecodedTrajectoryNapariPlotter class)

## Error Handling

All operations must fail gracefully:

- If Qt widgets cannot be imported → return None, log warning
- If `viewer.window` doesn't exist → return None, log warning  
- If `add_dock_widget()` fails → return None, log warning
- If widget creation fails → return None, log warning
- In `_log_to_console()`, if widget write fails → silently continue (stdout still works)

## Implementation Approach

1. Define `CustomLoggingWidget` class (can reuse/adapt `LogViewer` from codebase)
2. Add `custom_logging_widget` field to class definition