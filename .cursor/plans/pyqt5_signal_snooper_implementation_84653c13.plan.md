---
name: PyQt5 Signal Snooper Implementation
overview: Create a fully functional, non-invasive PyQt5 signal snooper that monitors signals from an object and its child widget tree, using callback functions to report emissions without affecting application behavior.
todos: []
---

# PyQt5 Signal Snooper Imple

mentation

## Overview

Create a non-invasive signal monitoring utility that can be passed a PyQt5 object and monitors all signal emissions from that object and its child widget tree. The snooper uses connection-based monitoring (connecting to signals rather than intercepting emit calls) to guarantee it doesn't affect application behavior.

## Implementation Details

### File Location

Create new file: `pyPhoCoreHelpers/src/pyphocorehelpers/gui/Qt/SignalSnooper.py`This location follows the existing pattern where Qt utilities are stored (similar to `ExceptionPrintingSlot.py`).

### Core Architecture

The snooper will use a **connection-based approach** for maximum safety:

- Instead of monkey-patching `Signal.emit()`, we connect a logging QObject to each signal
- This is non-invasive because we're only adding listeners, not modifying signal behavior
- Signal emissions proceed normally with no timing or argument changes
- Our logging happens asynchronously through Qt's signal/slot mechanism

### Key Components

1. **SignalSnooper Class** (`SignalSnooper`)

- Main class that takes a target object and optional callback
- Recursively discovers all QObject children in the widget tree
- Finds all signals on each object using introspection
- Connects logging slots to each signal
- Provides cleanup methods to disconnect all monitoring

2. **Signal Logger QObject** (internal helper)

- A QObject that provides logging slots
- Handles signal emissions safely with exception handling
- Calls user-provided callback with emission details

3. **Signal Discovery**

- Uses `inspect` module to find all Signal/pyqtSignal attributes
- Handles both `QtCore.Signal` (PySide-style) and `pyqtSignal` (PyQt-style)
- Recursively walks widget tree using `QObject.findChildren()`

### Safety Guarantees

1. **Exception Isolation**: All logging code wrapped in try/except to prevent exceptions from affecting application
2. **No Signal Modification**: We never modify signal objects or their emit methods
3. **No Timing Changes**: Logging happens through Qt's normal signal/slot mechanism
4. **Clean Disconnection**: Provides explicit cleanup to remove all connections
5. **Weak References**: Use weak references where appropriate to avoid circular dependencies

### API Design

```python
class SignalSnooper:
    def __init__(self, target_object, callback=None, include_children=True, 
                 signal_name_filter=None):
        """
        Args:
            target_object: QObject to monitor
            callback: Optional function(signal_name, object, args, kwargs) -> None
            include_children: If True, monitor child widgets recursively
            signal_name_filter: Optional list of signal names to monitor (None = all)
        """
    
    def start(self):
        """Begin monitoring signals"""
    
    def stop(self):
        """Stop monitoring and disconnect all logging connections"""
    
    def get_emission_count(self, signal_name=None):
        """Get count of emissions (optionally filtered by signal name)"""
    
    def clear_stats(self):
        """Clear emission statistics"""
```



### Implementation Strategy

1. **Signal Detection**: 

- Check if attribute is instance of `Signal` or `pyqtSignal`
- Handle both class-level and instance-level signals
- Filter out non-signal attributes safely

2. **Child Discovery**:

- Use `QObject.findChildren(QObject)` to get all child objects
- Filter to only QObject instances (not all Python objects)
- Handle cases where children might be added/removed dynamically

3. **Connection Management**:

- Store all connections in a list for easy cleanup
- Use `signal.connect()` to attach logging slots
- Handle connection failures gracefully (some signals might not be connectable)

4. **Callback Invocation**:

- Call user callback with: signal_name, emitting_object, args, kwargs, timestamp
- Wrap in try/except to ensure callback errors don't affect application
- Provide default console logging if no callback provided

### Edge Cases to Handle

- Objects without signals (skip gracefully)
- Signals that can't be connected (handle exceptions)
- Dynamic signal creation (only monitor signals present at start time)
- Circular references (use weak references for cleanup)
- Signals with different signatures (use `*args, **kwargs` in slots)
- Child objects being destroyed (handle disconnection gracefully)

### Testing Considerations

The implementation should:

- Work with both PyQt5 and PySide2 (via qtpy if available)
- Handle signals with various argument types
- Not interfere with normal signal/slot connections
- Clean up properly when stopped

## Files to Create

1. `pyPhoCoreHelpers/src/pyphocorehelpers/gui/Qt/SignalSnooper.py` - Main implementation