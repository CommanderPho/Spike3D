# Set PyQt5 as the preferred binding before importing silx
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

# Import Silx Qt
from silx.gui import qt
import sys
import subprocess
from pathlib import Path


class SilxExampleBrowser(qt.QMainWindow):
    """Interactive browser for Silx examples, similar to PyQtGraph's examples.run()"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Silx Examples Browser")
        self.setGeometry(100, 100, 1200, 800)
        
        # Get the examples directory path
        current_file = Path(__file__).resolve()
        examples_dir = current_file.parent / "silx_examples"
        self.examples_dir = examples_dir
        
        # Scan for examples
        self.examples = self.scan_examples()
        
        # Create UI
        self.create_ui()
        
        # Select first example if available
        if self.examples and self.example_list.count() > 0:
            self.example_list.setCurrentRow(0)
            self.on_example_selected()
    
    def scan_examples(self):
        """Scan the examples directory and return list of (name, path, description) tuples"""
        examples = []
        if not self.examples_dir.exists():
            return examples
        
        for py_file in sorted(self.examples_dir.glob("*.py")):
            # Skip __init__.py
            if py_file.name == "__init__.py":
                continue
            
            # Get module name without extension
            name = py_file.stem
            
            # Try to extract description from docstring
            description = self.get_example_description(py_file)
            
            examples.append((name, py_file, description))
        
        return examples
    
    def get_example_description(self, file_path):
        """Extract description from module docstring"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for module docstring
                if '"""' in content:
                    start = content.find('"""') + 3
                    end = content.find('"""', start)
                    if end > start:
                        docstring = content[start:end].strip()
                        # Get first line or first sentence
                        first_line = docstring.split('\n')[0]
                        return first_line[:100]  # Limit length
        except Exception:
            pass
        return "Silx example"
    
    def create_ui(self):
        """Create the user interface"""
        central_widget = qt.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = qt.QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Left panel: Example list
        left_panel = qt.QWidget()
        left_layout = qt.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label for example list
        list_label = qt.QLabel("Examples:")
        list_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        left_layout.addWidget(list_label)
        
        # List widget for examples
        self.example_list = qt.QListWidget()
        self.example_list.setMaximumWidth(300)
        for name, _, description in self.examples:
            item = qt.QListWidgetItem(name)
            item.setToolTip(description)
            self.example_list.addItem(item)
        self.example_list.itemSelectionChanged.connect(self.on_example_selected)
        left_layout.addWidget(self.example_list)
        
        # Run button
        self.run_button = qt.QPushButton("Run Example")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_example)
        left_layout.addWidget(self.run_button)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # Right panel: Code preview
        right_panel = qt.QWidget()
        right_layout = qt.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label for code preview
        code_label = qt.QLabel("Source Code:")
        code_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        right_layout.addWidget(code_label)
        
        # Description label
        self.description_label = qt.QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        right_layout.addWidget(self.description_label)
        
        # Text edit for code preview
        self.code_preview = qt.QPlainTextEdit()
        self.code_preview.setReadOnly(True)
        self.code_preview.setFont(qt.QFont("Consolas", 9) if sys.platform == "win32" else qt.QFont("Monospace", 9))
        right_layout.addWidget(self.code_preview)
        
        main_layout.addWidget(right_panel, stretch=1)
    
    def on_example_selected(self):
        """Handle example selection - load and display code"""
        current_item = self.example_list.currentItem()
        if not current_item:
            self.run_button.setEnabled(False)
            return
        
        # Find the selected example
        example_name = current_item.text()
        selected_example = None
        for name, path, description in self.examples:
            if name == example_name:
                selected_example = (name, path, description)
                break
        
        if not selected_example:
            return
        
        name, path, description = selected_example
        
        # Update description
        self.description_label.setText(f"<b>{name}</b><br>{description}")
        
        # Load and display code
        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            self.code_preview.setPlainText(code)
            self.run_button.setEnabled(True)
        except Exception as e:
            self.code_preview.setPlainText(f"Error loading file: {str(e)}")
            self.run_button.setEnabled(False)
    
    def run_example(self):
        """Execute the selected example in a subprocess"""
        current_item = self.example_list.currentItem()
        if not current_item:
            return
        
        example_name = current_item.text()
        selected_example = None
        for name, path, description in self.examples:
            if name == example_name:
                selected_example = (name, path, description)
                break
        
        if not selected_example:
            return
        
        name, path, description = selected_example
        
        # Run the example in a subprocess
        try:
            # Use the same Python interpreter
            python_exe = sys.executable
            script_path = str(path.resolve())
            
            # Run in subprocess to avoid QApplication conflicts
            subprocess.Popen([python_exe, script_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
        except Exception as e:
            qt.QMessageBox.critical(self, "Error", f"Failed to run example:\n{str(e)}")


def main():
    """Main entry point"""
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler
    
    browser = SilxExampleBrowser()
    browser.show()
    
    result = app.exec()
    app.deleteLater()
    sys.excepthook = sys.__excepthook__
    sys.exit(result)


if __name__ == '__main__':
    main()
