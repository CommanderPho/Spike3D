# Set PyQt5 as the preferred binding for any Qt-using examples
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import sys
import subprocess
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QRegExp, Qt, QSettings, QThread, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QSyntaxHighlighter, QTextCharFormat, QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QPlainTextEdit, QSplitter, QApplication, QMessageBox, QMenu)

from pyphoplacecellanalysis.GUI.Qt.Widgets.PhoCodeConsoleWidget import PhoCodeConsoleWidget


def _subprocess_python_executable() -> str:
    """Interpreter for spawning example processes. When a venv is active (prefix != base_prefix), use the python next to sys.prefix so we do not rely on a mis-set sys.executable (common on Windows / pythonw). Prefer python.exe over pythonw.exe for console subprocesses."""
    prefix = Path(sys.prefix).resolve()
    if sys.platform == "win32":
        venv_python = prefix / "Scripts" / "python.exe"
    else:
        venv_python = prefix / "bin" / "python"
    if sys.prefix != sys.base_prefix and venv_python.is_file():
        return str(venv_python.resolve())
    exe = Path(sys.executable).resolve()
    if sys.platform == "win32" and exe.name.lower() == "pythonw.exe":
        alt = exe.with_name("python.exe")
        if alt.is_file():
            return str(alt)
    return str(exe)


class ExampleSubprocessRunner(QThread):
    """Runs ``subprocess.Popen`` in a thread and streams merged stdout/stderr via ``chunk_ready``. Avoids QProcess (some Windows setups always fail with ``UnknownError`` even for ``cmd.exe``)."""

    chunk_ready = pyqtSignal(str)
    process_finished = pyqtSignal(int)

    def __init__(self, argv: list[str], cwd: str, parent: Any = None):
        super().__init__(parent)
        self._argv = argv
        self._cwd = cwd
        self._proc: Any = None

    def terminate_subprocess(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass

    def run(self) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        try:
            self._proc = subprocess.Popen(self._argv, cwd=self._cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=None, creationflags=creationflags)
        except Exception as e:
            self.chunk_ready.emit("Failed to start subprocess: %s\n" % (e,))
            self.process_finished.emit(-1)
            return
        out = self._proc.stdout
        assert out is not None
        try:
            while True:
                chunk = out.read(4096)
                if not chunk:
                    break
                self.chunk_ready.emit(chunk.decode("utf-8", errors="replace"))
        except Exception as e:
            self.chunk_ready.emit("\n[stdout read error] %s\n" % (e,))
        finally:
            try:
                out.close()
            except Exception:
                pass
            if self._proc.poll() is None:
                self._proc.wait()
            rc = self._proc.returncode if self._proc.returncode is not None else -1
            self.process_finished.emit(rc)


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
        self._favorites = set(self._load_favorites())
        self._example_process = None
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


    def _display_name(self, name):
        """Return list display text for an example name (with (*) prefix if favorite)."""
        return ("(*) " + name) if name in self._favorites else name


    def _load_favorites(self):
        """Load favorite example names from QSettings."""
        settings = QSettings("Spike3D", "VispyExampleBrowser")
        return settings.value("favorites", [], type=list) or []


    def _save_favorites(self):
        """Persist favorite example names to QSettings."""
        settings = QSettings("Spike3D", "VispyExampleBrowser")
        settings.setValue("favorites", list(self._favorites))


    def create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setSpacing(5)
        outer_layout.setContentsMargins(5, 5, 5, 5)

        top_area = QWidget()
        main_layout = QHBoxLayout(top_area)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        list_label = QLabel("Examples:")
        list_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        left_layout.addWidget(list_label)

        self.example_list = QListWidget()
        self.example_list.setMaximumWidth(300)
        self.example_list.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore[attr-defined]
        self.example_list.customContextMenuRequested.connect(self._on_list_context_menu)
        for name, _, description in self.examples:
            item = QListWidgetItem(self._display_name(name))
            item.setData(Qt.UserRole, name)  # type: ignore[attr-defined]
            item.setToolTip(description)
            self.example_list.addItem(item)
        self.example_list.itemSelectionChanged.connect(self.on_example_selected)
        left_layout.addWidget(self.example_list)

        self.run_button = QPushButton("Run Example")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.run_example)
        left_layout.addWidget(self.run_button)

        self.open_editor_button = QPushButton("Open in Default Editor")
        self.open_editor_button.setEnabled(False)
        self.open_editor_button.clicked.connect(self.open_in_editor)
        left_layout.addWidget(self.open_editor_button)

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

        here = Path(__file__).resolve().parent
        wrapper = here / "_run_vispy_example.py"
        sample_name = self.examples[0][0] if self.examples else "relative/path/to_example"
        sample_script = self.examples[0][1] if self.examples else (self.examples_dir / "your_example.py")
        win_hint = "On Windows, quote paths with spaces in ! lines; ! blocks the UI until the command exits."
        welcome_lines = [
            "PhoCodeConsoleWidget — vispy example output and commands",
            "",
            "Preferred (same subprocess as Run Example; does not block the browser UI):",
            "  browser.run_example()     # requires a selected list item",
            '  run_vispy_example("%s")   # run by scanned name' % (sample_name,),
            "",
            "This panel also shows merged stdout/stderr from the example subprocess (background thread).",
            "",
            "Advanced (block the Qt UI until the child script exits):",
            '  %%run "%s" "%s" "%s"' % (wrapper, sample_name, sample_script),
            "  run \"...\"  # same in-process runpy as %run",
            "  !\"%s\" \"%s\" \"%s\" \"%s\"   # shell; %s" % (_subprocess_python_executable(), wrapper, sample_name, sample_script, win_hint),
            "",
            "%run/run merge names into the console namespace; share one interpreter with this browser (vispy/Qt caveats).",
            "Shell ! is disabled if the widget is constructed with enable_shell_commands=False.",
            "",
        ]
        console_ns = {
            "browser": self,
            "vispy_examples_dir": self.examples_dir,
            "vispy_run_helper": wrapper,
            "vispy_python": _subprocess_python_executable(),
            "run_vispy_example": self.run_vispy_example,
        }
        self._run_console = PhoCodeConsoleWidget(parent=self, namespace=console_ns, text="\n".join(welcome_lines) + "\n", enable_shell_commands=True)
        self._run_console.setMinimumHeight(180)

        splitter = QSplitter(Qt.Vertical)  # type: ignore[attr-defined]
        splitter.addWidget(top_area)
        splitter.addWidget(self._run_console)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        outer_layout.addWidget(splitter)


    def closeEvent(self, event: QCloseEvent):
        if self._example_process is not None:
            runner = self._example_process
            self._example_process = None
            for _sig in (runner.chunk_ready, runner.process_finished):
                try:
                    _sig.disconnect()
                except TypeError:
                    pass
            runner.terminate_subprocess()
            runner.wait(60000)
            runner.deleteLater()
        super().closeEvent(event)


    def _stop_running_example_process(self):
        """Stop the current example subprocess without blocking the UI; thread ``finished`` deletes the runner."""
        if self._example_process is None:
            return
        runner = self._example_process
        self._example_process = None
        try:
            runner.chunk_ready.disconnect()
        except TypeError:
            pass
        try:
            runner.process_finished.disconnect()
        except TypeError:
            pass
        runner.terminate_subprocess()


    def _on_example_subprocess_finished(self, exit_code: int):
        runner = self.sender()
        if runner is not self._example_process:
            return
        self._run_console.write("\n[Process finished] exit code: %s\n" % (exit_code,))
        self._example_process = None


    def _on_list_context_menu(self, pos):
        item = self.example_list.itemAt(pos)
        if not item:
            return
        name = item.data(Qt.UserRole) or item.text().lstrip("(*) ")  # type: ignore[attr-defined]
        menu = QMenu(self)
        if name in self._favorites:
            action = menu.addAction("Remove from favorites")
        else:
            action = menu.addAction("Add to favorites")
        action = menu.exec_(self.example_list.mapToGlobal(pos))
        if not action:
            return
        if name in self._favorites:
            self._favorites.discard(name)
        else:
            self._favorites.add(name)
        self._save_favorites()
        item.setText(self._display_name(name))


    def _canonical_name(self, item):
        """Return the example name for an item (without (*) prefix)."""
        if item is None:
            return None
        name = item.data(Qt.UserRole)  # type: ignore[attr-defined]
        if name is not None:
            return name
        text = item.text()
        return text.lstrip("(*) ") if text.startswith("(*) ") else text


    def _start_example_process(self, name, path):
        """Start wrapper + example in a background thread via ``subprocess.Popen`` (same argv as ``!`` shell); inherits env from this process."""
        try:
            python_exe = _subprocess_python_executable()
            script_path = str(path.resolve())
            wrapper_path = str(Path(__file__).resolve().parent / "_run_vispy_example.py")
            work_dir = str(path.resolve().parent)
            py_path, wrap_path, sk_path = Path(python_exe), Path(wrapper_path), Path(script_path)
            if not py_path.is_file():
                self._run_console.write("Interpreter missing or not a file: %s\n" % (python_exe,))
                return False
            if not wrap_path.is_file():
                self._run_console.write("Wrapper missing or not a file: %s\n" % (wrapper_path,))
                return False
            if not sk_path.is_file():
                self._run_console.write("Example script missing or not a file: %s\n" % (script_path,))
                return False
            if not Path(work_dir).is_dir():
                self._run_console.write("Working directory missing or not a dir: %s\n" % (work_dir,))
                return False
            argv = [python_exe, wrapper_path, name, script_path]
            runner = ExampleSubprocessRunner(argv, work_dir, self)
            runner.chunk_ready.connect(self._run_console.write)
            runner.process_finished.connect(self._on_example_subprocess_finished)
            runner.finished.connect(runner.deleteLater)
            self._example_process = runner
            runner.start()
            return True
        except Exception as e:
            self._run_console.write(f"Failed to run example: {str(e)}\n")
            return False


    def run_vispy_example(self, name):
        """Run a scanned example by list id (e.g. ``basics/line``). Same non-blocking subprocess path as Run Example."""
        selected = None
        for n, path, _d in self.examples:
            if n == name:
                selected = (n, path)
                break
        if not selected:
            self._run_console.write("Unknown example name: %s\n" % (name,))
            return
        n, path = selected
        self._stop_running_example_process()
        self._run_console.write(f"\n{'=' * 60}\nRun example: {n}\n{'=' * 60}\n")
        self._start_example_process(n, path)


    def on_example_selected(self):
        current_item = self.example_list.currentItem()
        if not current_item:
            self.run_button.setEnabled(False)
            self.open_editor_button.setEnabled(False)
            return

        example_name = self._canonical_name(current_item)
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
            self.open_editor_button.setEnabled(True)
        except Exception as e:
            self.code_preview.setPlainText(f"Error loading file: {str(e)}")
            self.run_button.setEnabled(False)
            self.open_editor_button.setEnabled(False)


    def run_example(self):
        current_item = self.example_list.currentItem()
        if not current_item:
            return

        example_name = self._canonical_name(current_item)
        selected_example = None
        for name, path, description in self.examples:
            if name == example_name:
                selected_example = (name, path, description)
                break

        if not selected_example:
            return

        name, path, description = selected_example
        try:
            self._stop_running_example_process()
            self._run_console.write(f"\n{'=' * 60}\nRun example: {name}\n{'=' * 60}\n")
            self._start_example_process(name, path)
        except Exception as e:
            self._run_console.write(f"Failed to run example: {str(e)}\n")


    def open_in_editor(self):
        current_item = self.example_list.currentItem()
        if not current_item:
            return

        example_name = self._canonical_name(current_item)
        selected_example = None
        for name, path, description in self.examples:
            if name == example_name:
                selected_example = (name, path, description)
                break

        if not selected_example:
            return

        _, path, _ = selected_example
        try:
            path_str = str(path.resolve())
            if sys.platform == "win32":
                os.startfile(path_str)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path_str])
            else:
                subprocess.Popen(["xdg-open", path_str])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file in default editor:\n{str(e)}")


def main():
    app = QApplication([])
    browser = VispyExampleBrowser()
    browser.show()
    result = app.exec_()
    app.deleteLater()
    sys.exit(result)


if __name__ == '__main__':
    main()
