# Set PyQt5 as the preferred binding for any Qt-using examples
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import sys
import subprocess
from pathlib import Path

from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QPlainTextEdit, QApplication, QMessageBox)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code"""

    def __init__(self, parent):
        super().__init__(parent)

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))
        keyword_format.setFontWeight(700)

        string_format = QTextCharFormat()
        string_format.setForeground(QColor(206, 145, 120))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(106, 153, 85))
        comment_format.setFontItalic(True)

        function_format = QTextCharFormat()
        function_format.setForeground(QColor(220, 220, 170))
        function_format.setFontWeight(700)

        number_format = QTextCharFormat()
        number_format.setForeground(QColor(181, 206, 168))

        class_format = QTextCharFormat()
        class_format.setForeground(QColor(78, 201, 176))
        class_format.setFontWeight(700)

        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True',
            'try', 'while', 'with', 'yield'
        ]

        self.highlighting_rules = []
        for keyword in keywords:
            pattern = QRegExp(r'\b' + keyword + r'\b')
            self.highlighting_rules.append((pattern, keyword_format))

        string_patterns = [
            QRegExp(r'"[^"\\]*(\\.[^"\\]*)*"'),
            QRegExp(r"'[^'\\]*(\\.[^'\\]*)*'"),
            QRegExp(r'"""[^"]*"""'),
            QRegExp(r"'''[^']*'''"),
        ]
        for pattern in string_patterns:
            self.highlighting_rules.append((pattern, string_format))

        comment_pattern = QRegExp(r'#.*')
        self.highlighting_rules.append((comment_pattern, comment_format))

        function_pattern = QRegExp(r'\bdef\s+(\w+)\s*\(')
        self.highlighting_rules.append((function_pattern, function_format))

        class_pattern = QRegExp(r'\bclass\s+(\w+)')
        self.highlighting_rules.append((class_pattern, class_format))

        number_pattern = QRegExp(r'\b\d+\.?\d*\b')
        self.highlighting_rules.append((number_pattern, number_format))

    def highlightBlock(self, text):
        for pattern, fmt in self.highlighting_rules:
            index = pattern.indexIn(text)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, fmt)
                index = pattern.indexIn(text, index + length)


class VispyExampleBrowser(QMainWindow):
    """Interactive browser for Vispy examples, analogous to Silx Examples Browser"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vispy Examples Browser")
        self.setGeometry(100, 100, 1200, 800)

        current_file = Path(__file__).resolve()
        self.examples_dir = current_file.parent / "examples"
        self.examples = self.scan_examples()
        self.create_ui()

        if self.examples and self.example_list.count() > 0:
            self.example_list.setCurrentRow(0)
            self.on_example_selected()


    def scan_examples(self):
        """Scan the examples directory recursively and return list of (name, path, description) tuples"""
        examples = []
        if not self.examples_dir.exists():
            return examples

        for py_path in sorted(self.examples_dir.rglob("*.py")):
            if py_path.name == "__init__.py":
                continue
            try:
                rel = py_path.relative_to(self.examples_dir)
            except ValueError:
                continue
            name = str(rel.with_suffix("")).replace("\\", "/")
            description = self.get_example_description(py_path)
            examples.append((name, py_path, description))
        return examples


    def get_example_description(self, file_path):
        """Extract description from module docstring"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '"""' in content:
                    start = content.find('"""') + 3
                    end = content.find('"""', start)
                    if end > start:
                        docstring = content[start:end].strip()
                        first_line = docstring.split('\n')[0]
                        return first_line[:100]
        except Exception:
            pass
        return "Vispy example"


    def create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("Examples:")
        list_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        left_layout.addWidget(list_label)

        self.example_list = QListWidget()
        self.example_list.setMaximumWidth(300)
        for name, _, description in self.examples:
            item = QListWidgetItem(name)
            item.setToolTip(description)
            self.example_list.addItem(item)
        self.example_list.itemSelectionChanged.connect(self.on_example_selected)
        left_layout.addWidget(self.example_list)

        self.run_button = QPushButton("Run Example")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_example)
        left_layout.addWidget(self.run_button)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        code_label = QLabel("Source Code:")
        code_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        right_layout.addWidget(code_label)

        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        right_layout.addWidget(self.description_label)

        self.code_preview = QPlainTextEdit()
        self.code_preview.setReadOnly(True)
        self.code_preview.setFont(QFont("Consolas", 9) if sys.platform == "win32" else QFont("Monospace", 9))
        self.highlighter = PythonSyntaxHighlighter(self.code_preview.document())
        right_layout.addWidget(self.code_preview)

        main_layout.addWidget(right_panel, 1)


    def on_example_selected(self):
        current_item = self.example_list.currentItem()
        if not current_item:
            self.run_button.setEnabled(False)
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
        self.description_label.setText(f"<b>{name}</b><br>{description}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            self.code_preview.setPlainText(code)
            self.run_button.setEnabled(True)
        except Exception as e:
            self.code_preview.setPlainText(f"Error loading file: {str(e)}")
            self.run_button.setEnabled(False)


    def run_example(self):
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
        try:
            python_exe = sys.executable
            script_path = str(path.resolve())
            subprocess.Popen([python_exe, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run example:\n{str(e)}")


def main():
    app = QApplication([])
    browser = VispyExampleBrowser()
    browser.show()
    result = app.exec_()
    app.deleteLater()
    sys.exit(result)


if __name__ == '__main__':
    main()
