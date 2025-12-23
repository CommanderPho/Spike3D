---
name: Add flag to disable live window monitoring
overview: Add a configurable flag to enable/disable live window event interval monitoring in LiveWindowEventIntervalMonitoringMixin to improve rendering performance when monitoring isn't needed.
todos:
  - id: add_flag_init
    content: Add _enable_live_window_event_interval_monitoring flag initialization in LiveWindowEventIntervalMonitoringMixin_on_init() with default value True
    status: completed
  - id: add_property
    content: Add enable_live_window_event_interval_monitoring property with getter and setter after active_window_visible_intervals_dict property
    status: completed
    dependencies:
      - add_flag_init
  - id: add_flag_check
    content: Add flag check in LiveWindowEventIntervalMonitoringMixin_on_window_update() to skip on_visible_intervals_changed() when flag is False
    status: completed
    dependencies:
      - add_flag_init
---

# Add Flag to Enable/Disable Live Window Event Interval Monitoring

## Overview

Add a flag variable to `LiveWindowEventIntervalMonitoringMixin` that allows disabling the live window event interval monitoring feature. When disabled, the expensive `on_visible_intervals_changed()` computation will be skipped during window updates, improving rendering performance.

## Implementation Details

### File to Modify

- [`pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/EpochRenderingMixin.py`](pyphoplacecellanalysis/GUI/PyQtPlot/Widgets/Mixins/RenderTimeEpochs/EpochRenderingMixin.py)

### Changes

1. **Add flag initialization in `LiveWindowEventIntervalMonitoringMixin_on_init`** (around line 79-81):

- Add `self._enable_live_window_event_interval_monitoring = True` to initialize the flag
- Default to `True` to maintain backward compatibility (monitoring enabled by default)

2. **Add property for the flag** (after line 122):

- Create a property `enable_live_window_event_interval_monitoring` with getter and setter
- This allows external code to enable/disable monitoring at runtime

3. **Add flag check in `LiveWindowEventIntervalMonitoringMixin_on_window_update`** (around line 107-109):

- Before calling `self.on_visible_intervals_changed()`, check if `self._enable_live_window_event_interval_monitoring` is `True`
- Only execute the monitoring logic if the flag is enabled
- Early return if disabled to skip the expensive computation

### Code Locations

- **Flag initialization**: `LiveWindowEventIntervalMonitoringMixin_on_init()` method (line ~79-81)
- **Property definition**: After `active_window_visible_intervals_dict` property (line ~122)
- **Flag check**: `LiveWindowEventIntervalMonitoringMixin_on_window_update()` method (line ~107-109)