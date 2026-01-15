---
name: Create Silx Example Browser
overview: Create a Silx_EXAMPLES.py file that provides an interactive example browser similar to PyQtGraph's examples.run(), allowing users to browse, preview, and run Silx examples from the silx_examples directory.
todos: []
---

# Create Silx Example Browser

## Overview

Create `Silx_EXAMPLES.py` in `Spike3D/LibrariesExamples/Silx/` that provides an interactive browser for Silx examples, analogous to `PyQtPlot_EXAMPLES.py` for PyQtGraph.

## Implementation Details

### File Location

- **Target**: `Spike3D/LibrariesExamples/Silx/Silx_EXAMPLES.py`
- **Reference**: `Spike3D/LibrariesExamples/PyQtPlot/PyQtPlot_EXAMPLES.py`

### Key Components

1. **Example Browser Window**

   - Use `silx.gui.qt` (Silx's Qt wrapper) to create a QMainWindow
   - Split view: left panel for example list, right panel for code preview
   - Add "Run Example" button to execute selected example

2. **Example Discovery**

   - Scan `Spike3D/LibrariesExamples/Silx/silx_examples/` directory
   - Filter for `.py` files (exclude `__init__.py` and non-Python files)
   - Extract example names and descriptions from docstrings

3. **Code Preview**

   - Display source code of selected example in a QTextEdit or QPlainTextEdit
   - Syntax highlighting (optional, if available)
   - Read-only view

4. **Example Execution**

   - Use `subprocess` or `importlib` to run selected examples
   - Handle QApplication conflicts (examples create their own QApplication)
   - Option: Run in separate process to avoid conflicts

5. **Main Entry Point**

   - Create QApplication instance
   - Show browser window
   - Execute Qt event loop with `app.exec()`

### Implementation Approach

**Option A: Import and Run (Simpler)**

- Dynamically import example modules
- Extract and call their `main()` function
- Risk: QApplication conflicts if examples create their own

**Option B: Subprocess Execution (Safer)**

- Run examples in separate Python processes
- Avoids QApplication conflicts
- More robust but slightly more complex

**Recommended**: Option B (subprocess) for better isolation and to avoid QApplication conflicts.

### Code Structure

```python
# Set PyQt5 as preferred binding (if needed)
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'  # May not be needed for Silx

# Import Silx Qt
from silx.gui import qt
import sys
import subprocess
import importlib.util
from pathlib import Path

class SilxExampleBrowser(qt.QMainWindow):
    def __init__(self):
        # Initialize UI
        # Scan examples directory
        # Create list widget
        # Create code preview
        # Create run button
    
    def scan_examples(self):
        # Find all .py files in silx_examples/
        # Return list of (name, path, description)
    
    def on_example_selected(self):
        # Load and display code
    
    def run_example(self):
        # Execute selected example in subprocess

def main():
    app = qt.QApplication([])
    browser = SilxExampleBrowser()
    browser.show()
    app.exec()

if __name__ == '__main__':
    main()
```

### Files to Create/Modify

- **Create**: `Spike3D/LibrariesExamples/Silx/Silx_EXAMPLES.py`
- **No modifications** needed to existing example files

### Considerations

- Handle examples that require command-line arguments (e.g., `hdf5widget.py`)
- Filter out non-runnable files (e.g., `fft.png`, `__init__.py`)
- Extract example descriptions from module docstrings for better UX
- Ensure proper path resolution for example files
- Handle errors gracefully when examples fail to run