#!/usr/bin/env python
"""
Interactive Matplotlib Mosaic Designer

A standalone GUI tool that lets you visually design subplot_mosaic layouts,
dynamically add/remove rows/columns, rename and merge cells, adjust ratios
and spacing, and generate copyable Python code.

Usage:
    python mosaic_designer.py

Dependencies:
    - matplotlib
    - tkinter (standard library)
"""
from __future__ import annotations
import ast
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import copy
import colorsys
import string
from typing import Optional, Union, List, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class MosaicModel:
    """Internal representation of a subplot_mosaic grid."""

    def __init__(self, rows: int = 2, cols: int = 2):
        self.rows = rows
        self.cols = cols
        self.grid: list[list[str]] = []
        self.height_ratios: list[float] = [1.0] * rows
        self.width_ratios: list[float] = [1.0] * cols
        self.wspace: float = 0.1
        self.hspace: float = 0.1
        self.fig_width: float = 12.0
        self.fig_height: float = 10.0
        self.empty_sentinel: str = "."
        self._counter = 0
        self._init_grid()

    # -- helpers ----------------------------------------------------------

    def _next_name(self) -> str:
        """Generate a unique axis name like ax0, ax1, …"""
        name = f"ax{self._counter}"
        self._counter += 1
        return name

    def _init_grid(self):
        self.grid = []
        for _r in range(self.rows):
            row = []
            for _c in range(self.cols):
                row.append(self._next_name())
            self.grid.append(row)

    # -- grid mutations ----------------------------------------------------

    def add_row(self):
        new_row = [self._next_name() for _ in range(self.cols)]
        self.grid.append(new_row)
        self.rows += 1
        self.height_ratios.append(1.0)

    def remove_row(self):
        if self.rows <= 1:
            return
        self.grid.pop()
        self.rows -= 1
        self.height_ratios.pop()

    def add_column(self):
        for row in self.grid:
            row.append(self._next_name())
        self.cols += 1
        self.width_ratios.append(1.0)

    def remove_column(self):
        if self.cols <= 1:
            return
        for row in self.grid:
            row.pop()
        self.cols -= 1
        self.width_ratios.pop()

    def rename_cell(self, old_name: str, new_name: str):
        """Rename *all* occurrences of *old_name* to *new_name*."""
        if new_name == old_name:
            return
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == old_name:
                    self.grid[r][c] = new_name

    def set_cell(self, r: int, c: int, name: str):
        self.grid[r][c] = name

    def toggle_empty(self, r: int, c: int):
        """Toggle a cell between its name and the empty sentinel."""
        if self.grid[r][c] == self.empty_sentinel:
            self.grid[r][c] = self._next_name()
        else:
            self.grid[r][c] = self.empty_sentinel

    def merge_cells(self, r1: int, c1: int, r2: int, c2: int):
        """Set all cells in the rectangle [r1..r2, c1..c2] to the same name."""
        name = self.grid[r1][c1]
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.grid[r][c] = name

    def split_cell(self, name: str):
        """Give every occurrence of *name* a unique new name."""
        first = True
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == name:
                    if first:
                        first = False  # keep the first occurrence
                    else:
                        self.grid[r][c] = self._next_name()

    def unique_names(self) -> list[str]:
        """Return unique cell names preserving first-appearance order."""
        seen = set()
        result = []
        for row in self.grid:
            for name in row:
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        return result

    def cell_span(self, name: str):
        """Return (min_row, min_col, max_row, max_col) for a named cell."""
        coords = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid[r][c] == name]
        if not coords:
            return None
        rows = [rc[0] for rc in coords]
        cols = [rc[1] for rc in coords]
        return min(rows), min(cols), max(rows), max(cols)

    def is_valid_mosaic(self) -> tuple[bool, str]:
        """Check if the current grid is a valid subplot_mosaic layout.
        Each named region must form a filled rectangle.
        """
        for name in self.unique_names():
            if name == self.empty_sentinel:
                continue
            span = self.cell_span(name)
            if span is None:
                continue
            r1, c1, r2, c2 = span
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if self.grid[r][c] != name:
                        return False, f'Region "{name}" is not a filled rectangle.'
        return True, ""

    # -- code generation ---------------------------------------------------

    def generate_code(self) -> str:
        valid, msg = self.is_valid_mosaic()
        if not valid:
            return f"# ⚠ Invalid mosaic layout: {msg}\n# Fix the grid before generating code."

        indent = "    "
        lines: list[str] = []
        lines.append("import matplotlib.pyplot as plt")
        lines.append("")
        lines.append(f"fig = plt.figure(figsize=({self.fig_width}, {self.fig_height}))")
        lines.append("")

        # mosaic_layout
        lines.append("mosaic_layout = [")
        for row in self.grid:
            row_str = ", ".join(f'"{n}"' for n in row)
            lines.append(f"{indent}[{row_str}],")
        lines.append("]")
        lines.append("")

        # build subplot_mosaic call
        lines.append("ax_dict = fig.subplot_mosaic(")
        lines.append(f"{indent}mosaic_layout,")

        # empty_sentinel (only if non-default used)
        uses_empty = any(self.grid[r][c] == self.empty_sentinel
                         for r in range(self.rows) for c in range(self.cols))
        if uses_empty and self.empty_sentinel != ".":
            lines.append(f'{indent}empty_sentinel="{self.empty_sentinel}",')

        # height_ratios
        hr = [_fmt_num(v) for v in self.height_ratios]
        lines.append(f"{indent}height_ratios=[{', '.join(hr)}],")

        # width_ratios
        if self.cols > 1:
            wr = [_fmt_num(v) for v in self.width_ratios]
            lines.append(f"{indent}width_ratios=[{', '.join(wr)}],")

        # gridspec_kw
        lines.append(f"{indent}gridspec_kw=dict(wspace={_fmt_num(self.wspace)}, hspace={_fmt_num(self.hspace)}),")
        lines.append(")")
        lines.append("")
        lines.append("plt.tight_layout()")
        lines.append("plt.show()")
        return "\n".join(lines)

    def load_from_code(self, code_str: str) -> tuple[bool, str]:
        """Parse Python code containing a subplot_mosaic definition and update the model."""
        try:
            tree = ast.parse(code_str)
        except Exception as e:
            return False, f"Syntax Error: {e}"

        extracted_grid = None
        extracted_empty_sentinel = "."
        extracted_height_ratios = None
        extracted_width_ratios = None
        extracted_wspace = None
        extracted_hspace = None
        extracted_fig_width = None
        extracted_fig_height = None

        var_dict = {}

        def eval_node(node):
            if node is None:
                return None
            try:
                return ast.literal_eval(node)
            except Exception:
                return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        val = eval_node(node.value)
                        if val is not None:
                            var_dict[target.id] = val

            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name in ("figure", "Figure"):
                    for kw in node.keywords:
                        if kw.arg == "figsize":
                            val = eval_node(kw.value)
                            if isinstance(val, (tuple, list)) and len(val) == 2:
                                try:
                                    extracted_fig_width, extracted_fig_height = float(val[0]), float(val[1])
                                except (ValueError, TypeError):
                                    pass

                if func_name == "subplot_mosaic":
                    layout_node = None
                    if node.args:
                        layout_node = node.args[0]
                    else:
                        for kw in node.keywords:
                            if kw.arg in ("mosaic", "layout"):
                                layout_node = kw.value

                    if layout_node is not None:
                        if isinstance(layout_node, ast.Name) and layout_node.id in var_dict:
                            extracted_grid = var_dict[layout_node.id]
                        else:
                            extracted_grid = eval_node(layout_node)

                    for kw in node.keywords:
                        val = eval_node(kw.value)
                        if val is None and isinstance(kw.value, ast.Name) and kw.value.id in var_dict:
                            val = var_dict[kw.value.id]

                        if kw.arg == "empty_sentinel":
                            if isinstance(val, str):
                                extracted_empty_sentinel = val
                        elif kw.arg == "height_ratios":
                            if isinstance(val, (tuple, list)):
                                try:
                                    extracted_height_ratios = [float(v) for v in val]
                                except (ValueError, TypeError):
                                    pass
                        elif kw.arg == "width_ratios":
                            if isinstance(val, (tuple, list)):
                                try:
                                    extracted_width_ratios = [float(v) for v in val]
                                except (ValueError, TypeError):
                                    pass
                        elif kw.arg == "gridspec_kw":
                            gkw = {}
                            if isinstance(kw.value, ast.Dict):
                                gkw = eval_node(kw.value) or {}
                            elif isinstance(kw.value, ast.Call):
                                for gkw_item in kw.value.keywords:
                                    gval = eval_node(gkw_item.value)
                                    if gval is not None:
                                        gkw[gkw_item.arg] = gval
                            if isinstance(gkw, dict):
                                if "wspace" in gkw:
                                    try:
                                        extracted_wspace = float(gkw["wspace"])
                                    except (ValueError, TypeError):
                                        pass
                                if "hspace" in gkw:
                                    try:
                                        extracted_hspace = float(gkw["hspace"])
                                    except (ValueError, TypeError):
                                        pass

        if extracted_grid is None:
            for k in ("mosaic_layout", "mosaic", "layout"):
                if k in var_dict:
                    extracted_grid = var_dict[k]
                    break

        if extracted_grid is None:
            return False, "Could not find a mosaic layout variable or subplot_mosaic call in the code."

        formatted_grid = []
        if isinstance(extracted_grid, (list, tuple)):
            for row in extracted_grid:
                if isinstance(row, (list, tuple)):
                    formatted_grid.append([str(item) for item in row])
                elif isinstance(row, str):
                    if " " in row.strip():
                        formatted_grid.append(row.strip().split())
                    elif ";" in row:
                        formatted_grid.append([x.strip() for x in row.split(";")])
                    else:
                        formatted_grid.append(list(row))
                else:
                    return False, f"Invalid layout row format: {row}"
        elif isinstance(extracted_grid, str):
            lines = [line.strip() for line in extracted_grid.strip().splitlines() if line.strip()]
            for line in lines:
                if " " in line:
                    formatted_grid.append(line.split())
                else:
                    formatted_grid.append(list(line))

        if not formatted_grid:
            return False, "Mosaic layout grid is empty."

        cols = len(formatted_grid[0])
        if cols == 0:
            return False, "Mosaic layout grid has 0 columns."
        for r_idx, row in enumerate(formatted_grid):
            if len(row) != cols:
                return False, f"Row {r_idx} has {len(row)} items, but Row 0 has {cols} items. All rows in mosaic_layout must have the same length."

        rows = len(formatted_grid)

        # Update model properties
        self.rows = rows
        self.cols = cols
        self.grid = formatted_grid
        self.empty_sentinel = extracted_empty_sentinel

        if extracted_height_ratios and len(extracted_height_ratios) == rows:
            self.height_ratios = extracted_height_ratios
        else:
            self.height_ratios = [1.0] * rows

        if extracted_width_ratios and len(extracted_width_ratios) == cols:
            self.width_ratios = extracted_width_ratios
        else:
            self.width_ratios = [1.0] * cols

        if extracted_wspace is not None:
            self.wspace = extracted_wspace
        if extracted_hspace is not None:
            self.hspace = extracted_hspace
        if extracted_fig_width is not None:
            self.fig_width = extracted_fig_width
        if extracted_fig_height is not None:
            self.fig_height = extracted_fig_height

        return True, ""


def _fmt_num(v: float) -> str:
    """Format a float nicely (strip trailing zeros)."""
    if v == int(v):
        return str(int(v))
    return f"{v:.4g}"


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

def _generate_palette(n: int) -> list[str]:
    """Return *n* distinct pastel hex colours."""
    colors = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hls_to_rgb(h, 0.78, 0.55)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


# ---------------------------------------------------------------------------
# GUI application
# ---------------------------------------------------------------------------

class MosaicDesigner:
    """Main tkinter application."""

    CELL_MIN_SIZE = 50  # minimum pixel size for a grid cell in the preview
    PREVIEW_PADDING = 20

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Matplotlib Mosaic Designer")
        self.master.minsize(900, 750)

        self.model = MosaicModel(rows=2, cols=2)
        self._selected_cell: Optional[str] = None
        self._merge_anchor: Optional[Tuple[int, int]] = None  # for merge selection
        self._is_syncing_from_code = False

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Use a PanedWindow for resizable top/bottom split
        self._paned = ttk.PanedWindow(self.master, orient=tk.VERTICAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        # ---- Top: preview canvas ----
        top_frame = ttk.Frame(self._paned)
        self._paned.add(top_frame, weight=3)

        self._canvas = tk.Canvas(top_frame, bg="#2b2b2b", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", lambda e: self._draw_preview())
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas.bind("<Shift-Button-1>", self._on_canvas_shift_click)
        self._canvas.bind("<Button-3>", self._on_canvas_right_click)

        # ---- Bottom: controls + code ----
        bottom_frame = ttk.Frame(self._paned)
        self._paned.add(bottom_frame, weight=2)

        # ---- controls row ----
        ctrl_frame = ttk.LabelFrame(bottom_frame, text="Controls")
        ctrl_frame.pack(fill=tk.X, padx=6, pady=(6, 0))

        # -- grid buttons --
        grid_frame = ttk.LabelFrame(ctrl_frame, text="Grid")
        grid_frame.pack(side=tk.LEFT, padx=4, pady=4)

        ttk.Button(grid_frame, text="+ Row", command=self._add_row).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(grid_frame, text="− Row", command=self._remove_row).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(grid_frame, text="+ Col", command=self._add_col).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(grid_frame, text="− Col", command=self._remove_col).grid(row=1, column=1, padx=2, pady=2)

        # -- cell operations --
        cell_frame = ttk.LabelFrame(ctrl_frame, text="Cell")
        cell_frame.pack(side=tk.LEFT, padx=4, pady=4)

        ttk.Label(cell_frame, text="Name:").grid(row=0, column=0, padx=2, pady=2)
        self._name_var = tk.StringVar()
        self._name_entry = ttk.Entry(cell_frame, textvariable=self._name_var, width=14)
        self._name_entry.grid(row=0, column=1, padx=2, pady=2)
        self._name_entry.bind("<Return>", self._on_rename)

        ttk.Button(cell_frame, text="Rename", command=self._on_rename).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(cell_frame, text="Split", command=self._on_split).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(cell_frame, text="Toggle Empty", command=self._on_toggle_empty).grid(row=1, column=1, columnspan=2, padx=2, pady=2)

        # -- merge info --
        merge_frame = ttk.LabelFrame(ctrl_frame, text="Merge")
        merge_frame.pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Label(merge_frame, text="Click first cell,\nthen Shift+Click\nlast cell to merge.", justify=tk.LEFT).pack(padx=4, pady=4)

        # -- figure / spacing --
        fig_frame = ttk.LabelFrame(ctrl_frame, text="Figure")
        fig_frame.pack(side=tk.LEFT, padx=4, pady=4)

        ttk.Label(fig_frame, text="Width:").grid(row=0, column=0, padx=2, pady=1, sticky=tk.W)
        self._fw_var = tk.DoubleVar(value=self.model.fig_width)
        ttk.Spinbox(fig_frame, from_=2, to=40, increment=0.5, textvariable=self._fw_var, width=5,
                     command=self._on_fig_change).grid(row=0, column=1, padx=2)

        ttk.Label(fig_frame, text="Height:").grid(row=1, column=0, padx=2, pady=1, sticky=tk.W)
        self._fh_var = tk.DoubleVar(value=self.model.fig_height)
        ttk.Spinbox(fig_frame, from_=2, to=40, increment=0.5, textvariable=self._fh_var, width=5,
                     command=self._on_fig_change).grid(row=1, column=1, padx=2)

        ttk.Label(fig_frame, text="wspace:").grid(row=0, column=2, padx=2, pady=1, sticky=tk.W)
        self._ws_var = tk.DoubleVar(value=self.model.wspace)
        ttk.Spinbox(fig_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self._ws_var, width=5,
                     command=self._on_fig_change).grid(row=0, column=3, padx=2)

        ttk.Label(fig_frame, text="hspace:").grid(row=1, column=2, padx=2, pady=1, sticky=tk.W)
        self._hs_var = tk.DoubleVar(value=self.model.hspace)
        ttk.Spinbox(fig_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self._hs_var, width=5,
                     command=self._on_fig_change).grid(row=1, column=3, padx=2)

        # ---- ratios row ----
        self._ratio_frame = ttk.LabelFrame(bottom_frame, text="Ratios")
        self._ratio_frame.pack(fill=tk.X, padx=6, pady=4)

        # ---- code output ----
        code_frame = ttk.LabelFrame(bottom_frame, text="Generated Code")
        code_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        code_top = ttk.Frame(code_frame)
        code_top.pack(fill=tk.X)
        ttk.Button(code_top, text="📋 Copy to Clipboard", command=self._copy_code).pack(side=tk.RIGHT, padx=4, pady=2)
        ttk.Button(code_top, text="⚡ Apply Code to GUI", command=self._apply_code_to_gui).pack(side=tk.RIGHT, padx=4, pady=2)

        self._status_label = ttk.Label(code_top, text="Ctrl+Enter to apply code changes", font=("Arial", 9, "italic"), foreground="#888888")
        self._status_label.pack(side=tk.LEFT, padx=6, pady=2)

        self._code_text = tk.Text(code_frame, wrap=tk.NONE, height=10, bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="#d4d4d4", relief=tk.FLAT, padx=8, pady=6)
        self._code_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._code_text.bind("<Control-Return>", lambda e: (self._apply_code_to_gui(), "break")[1])
        self._code_text.bind("<Command-Return>", lambda e: (self._apply_code_to_gui(), "break")[1])

        # configure code font
        try:
            code_font = tkfont.Font(family="Consolas", size=10)
        except Exception:
            code_font = tkfont.Font(family="Courier", size=10)
        self._code_text.configure(font=code_font)

        # scrollbar
        sb = ttk.Scrollbar(self._code_text, orient=tk.VERTICAL, command=self._code_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._code_text.configure(yscrollcommand=sb.set)

        sbh = ttk.Scrollbar(self._code_text, orient=tk.HORIZONTAL, command=self._code_text.xview)
        sbh.pack(side=tk.BOTTOM, fill=tk.X)
        self._code_text.configure(xscrollcommand=sbh.set)

    # ------------------------------------------------------------------ ratio widgets
    def _rebuild_ratio_widgets(self):
        for w in self._ratio_frame.winfo_children():
            w.destroy()

        # height ratios
        ttk.Label(self._ratio_frame, text="Height ratios:").grid(row=0, column=0, padx=4, sticky=tk.W)
        self._hr_vars: list[tk.DoubleVar] = []
        for i in range(self.model.rows):
            var = tk.DoubleVar(value=self.model.height_ratios[i])
            self._hr_vars.append(var)
            ttk.Label(self._ratio_frame, text=f"R{i}:").grid(row=0, column=1 + i * 2, padx=(4, 0))
            ttk.Spinbox(self._ratio_frame, from_=0.1, to=50, increment=0.5, textvariable=var, width=5,
                         command=self._on_ratio_change).grid(row=0, column=2 + i * 2, padx=(0, 4))

        # width ratios
        ttk.Label(self._ratio_frame, text="Width ratios:").grid(row=1, column=0, padx=4, sticky=tk.W)
        self._wr_vars: list[tk.DoubleVar] = []
        for i in range(self.model.cols):
            var = tk.DoubleVar(value=self.model.width_ratios[i])
            self._wr_vars.append(var)
            ttk.Label(self._ratio_frame, text=f"C{i}:").grid(row=1, column=1 + i * 2, padx=(4, 0))
            ttk.Spinbox(self._ratio_frame, from_=0.1, to=50, increment=0.5, textvariable=var, width=5,
                         command=self._on_ratio_change).grid(row=1, column=2 + i * 2, padx=(0, 4))

    # ------------------------------------------------------------------ preview
    def _draw_preview(self):
        """Redraw the grid preview on the canvas."""
        self._canvas.delete("all")

        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        pad = self.PREVIEW_PADDING
        avail_w = cw - 2 * pad
        avail_h = ch - 2 * pad

        # compute cell sizes proportional to ratios
        total_hr = sum(self.model.height_ratios)
        total_wr = sum(self.model.width_ratios)

        col_widths = [(r / total_wr) * avail_w for r in self.model.width_ratios]
        row_heights = [(r / total_hr) * avail_h for r in self.model.height_ratios]

        # build colour map
        names = self.model.unique_names()
        non_empty = [n for n in names if n != self.model.empty_sentinel]
        palette = _generate_palette(len(non_empty))
        color_map = {n: palette[i] for i, n in enumerate(non_empty)}
        color_map[self.model.empty_sentinel] = "#3c3c3c"

        # store cell rectangles for hit testing
        self._cell_rects: list[tuple[int, int, float, float, float, float]] = []  # (r, c, x1, y1, x2, y2)

        drawn_spans: set[str] = set()

        for r in range(self.model.rows):
            for c in range(self.model.cols):
                name = self.model.grid[r][c]

                x1 = pad + sum(col_widths[:c])
                y1 = pad + sum(row_heights[:r])
                x2 = x1 + col_widths[c]
                y2 = y1 + row_heights[r]

                self._cell_rects.append((r, c, x1, y1, x2, y2))

                if name in drawn_spans:
                    continue

                # for spanning cells, compute the full rectangle
                span = self.model.cell_span(name)
                if span is not None:
                    sr1, sc1, sr2, sc2 = span
                    sx1 = pad + sum(col_widths[:sc1])
                    sy1 = pad + sum(row_heights[:sr1])
                    sx2 = sx1 + sum(col_widths[sc1:sc2 + 1])
                    sy2 = sy1 + sum(row_heights[sr1:sr2 + 1])
                else:
                    sx1, sy1, sx2, sy2 = x1, y1, x2, y2

                fill = color_map.get(name, "#888888")
                outline = "#ffffff" if name == self._selected_cell else "#555555"
                width = 3 if name == self._selected_cell else 1

                self._canvas.create_rectangle(sx1, sy1, sx2, sy2,
                                               fill=fill, outline=outline, width=width,
                                               tags=("cell",))

                # label
                cx = (sx1 + sx2) / 2
                cy = (sy1 + sy2) / 2
                font_size = max(9, min(16, int((sx2 - sx1) / max(len(name), 1) * 0.7)))
                self._canvas.create_text(cx, cy, text=name,
                                          fill="#222222", font=("Arial", font_size, "bold"),
                                          tags=("label",))

                if name != self.model.empty_sentinel:
                    drawn_spans.add(name)

    def _hit_test(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Return (row, col) for the cell under (x, y) or None."""
        for r, c, x1, y1, x2, y2 in self._cell_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return r, c
        return None

    # ------------------------------------------------------------------ events
    def _on_canvas_click(self, event):
        hit = self._hit_test(event.x, event.y)
        if hit is None:
            self._selected_cell = None
            self._merge_anchor = None
        else:
            r, c = hit
            name = self.model.grid[r][c]
            self._selected_cell = name
            self._merge_anchor = (r, c)
            self._name_var.set(name)
        self._draw_preview()

    def _on_canvas_shift_click(self, event):
        """Shift-click to complete a merge selection."""
        if self._merge_anchor is None:
            return
        hit = self._hit_test(event.x, event.y)
        if hit is None:
            return
        r2, c2 = hit
        r1, c1 = self._merge_anchor
        # normalise
        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)
        self.model.merge_cells(rmin, cmin, rmax, cmax)
        self._merge_anchor = None
        self._refresh()

    def _on_canvas_right_click(self, event):
        """Right-click to toggle empty."""
        hit = self._hit_test(event.x, event.y)
        if hit is None:
            return
        r, c = hit
        self.model.toggle_empty(r, c)
        self._refresh()

    def _add_row(self):
        self.model.add_row()
        self._refresh()

    def _remove_row(self):
        self.model.remove_row()
        self._refresh()

    def _add_col(self):
        self.model.add_column()
        self._refresh()

    def _remove_col(self):
        self.model.remove_column()
        self._refresh()

    def _on_rename(self, event=None):
        if self._selected_cell is None:
            return
        new_name = self._name_var.get().strip()
        if not new_name:
            messagebox.showwarning("Invalid name", "Cell name cannot be empty.")
            return
        # Check for name collisions (different region with same name)
        if new_name != self._selected_cell and new_name in self.model.unique_names():
            messagebox.showwarning("Duplicate name", f'Name "{new_name}" is already used by another cell.')
            return
        self.model.rename_cell(self._selected_cell, new_name)
        self._selected_cell = new_name
        self._refresh()

    def _on_split(self):
        if self._selected_cell is None:
            return
        self.model.split_cell(self._selected_cell)
        self._selected_cell = None
        self._refresh()

    def _on_toggle_empty(self):
        if self._selected_cell is None:
            return
        # find first occurrence
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.grid[r][c] == self._selected_cell:
                    self.model.toggle_empty(r, c)
                    self._selected_cell = self.model.grid[r][c]
                    self._refresh()
                    return

    def _on_fig_change(self):
        if getattr(self, "_is_syncing_from_code", False):
            return
        try:
            self.model.fig_width = self._fw_var.get()
        except (tk.TclError, ValueError):
            pass
        try:
            self.model.fig_height = self._fh_var.get()
        except (tk.TclError, ValueError):
            pass
        try:
            self.model.wspace = self._ws_var.get()
        except (tk.TclError, ValueError):
            pass
        try:
            self.model.hspace = self._hs_var.get()
        except (tk.TclError, ValueError):
            pass
        self._update_code()

    def _on_ratio_change(self):
        if getattr(self, "_is_syncing_from_code", False):
            return
        try:
            for i, var in enumerate(self._hr_vars):
                self.model.height_ratios[i] = max(0.1, var.get())
        except (tk.TclError, ValueError):
            pass
        try:
            for i, var in enumerate(self._wr_vars):
                self.model.width_ratios[i] = max(0.1, var.get())
        except (tk.TclError, ValueError):
            pass
        self._draw_preview()
        self._update_code()

    def _copy_code(self):
        code = self._code_text.get("1.0", tk.END).strip()
        self.master.clipboard_clear()
        self.master.clipboard_append(code)
        messagebox.showinfo("Copied", "Code copied to clipboard!")

    def _apply_code_to_gui(self, event=None):
        """Parse the Python code from the text area and update the GUI layout/model."""
        code = self._code_text.get("1.0", tk.END).strip()
        if not code:
            return

        ok, err_msg = self.model.load_from_code(code)
        if not ok:
            messagebox.showerror("Code Parsing Error", f"Could not update GUI layout from code:\n\n{err_msg}")
            return

        self._is_syncing_from_code = True
        try:
            self._fw_var.set(self.model.fig_width)
            self._fh_var.set(self.model.fig_height)
            self._ws_var.set(self.model.wspace)
            self._hs_var.set(self.model.hspace)
            self._selected_cell = None
            self._merge_anchor = None
            self._rebuild_ratio_widgets()
            self._draw_preview()
            self._highlight_code()
            if hasattr(self, "_status_label") and self._status_label:
                self._status_label.config(text="✓ GUI updated from code!", foreground="#4ec9b0")
                self.master.after(2500, lambda: self._status_label.config(text="Ctrl+Enter to apply code changes", foreground="#888888"))
        finally:
            self._is_syncing_from_code = False

    # ------------------------------------------------------------------ refresh
    def _refresh(self):
        """Full UI refresh."""
        self._rebuild_ratio_widgets()
        self._draw_preview()
        self._update_code()

    def _update_code(self):
        if getattr(self, "_is_syncing_from_code", False):
            return
        code = self.model.generate_code()
        self._code_text.configure(state=tk.NORMAL)
        self._code_text.delete("1.0", tk.END)
        self._code_text.insert("1.0", code)
        # simple syntax highlighting
        self._highlight_code()

    def _highlight_code(self):
        """Apply simple colour tags to the code text widget."""
        # configure tags
        self._code_text.tag_configure("keyword", foreground="#569cd6")
        self._code_text.tag_configure("string", foreground="#ce9178")
        self._code_text.tag_configure("number", foreground="#b5cea8")
        self._code_text.tag_configure("comment", foreground="#6a9955")
        self._code_text.tag_configure("builtin", foreground="#dcdcaa")

        content = self._code_text.get("1.0", tk.END)

        keywords = {"import", "from", "as", "def", "class", "return", "if", "else", "for", "in", "True", "False", "None"}
        builtins_set = {"dict", "list", "print", "range", "len"}

        for i, line in enumerate(content.split("\n"), start=1):
            # comments
            if "#" in line:
                cidx = line.index("#")
                self._code_text.tag_add("comment", f"{i}.{cidx}", f"{i}.end")

            for token in line.replace("(", " ").replace(")", " ").replace(",", " ").replace("=", " ").split():
                if token in keywords:
                    start = line.find(token)
                    if start >= 0:
                        self._code_text.tag_add("keyword", f"{i}.{start}", f"{i}.{start + len(token)}")
                elif token in builtins_set:
                    start = line.find(token)
                    if start >= 0:
                        self._code_text.tag_add("builtin", f"{i}.{start}", f"{i}.{start + len(token)}")

        # highlight strings
        import re
        for match in re.finditer(r'"[^"]*"', content):
            start_idx = match.start()
            end_idx = match.end()
            # convert to text widget index
            start_line, start_col = self._offset_to_index(content, start_idx)
            end_line, end_col = self._offset_to_index(content, end_idx)
            self._code_text.tag_add("string", f"{start_line}.{start_col}", f"{end_line}.{end_col}")

    @staticmethod
    def _offset_to_index(text: str, offset: int) -> tuple[int, int]:
        """Convert a character offset to (line, col) for a Text widget (1-indexed lines)."""
        line = 1
        col = 0
        for i, ch in enumerate(text):
            if i == offset:
                return line, col
            if ch == "\n":
                line += 1
                col = 0
            else:
                col += 1
        return line, col


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    # Try to apply a modern theme
    try:
        style = ttk.Style()
        available = style.theme_names()
        for preferred in ("clam", "alt", "vista", "xpnative"):
            if preferred in available:
                style.theme_use(preferred)
                break
    except Exception:
        pass

    app = MosaicDesigner(root)
    root.mainloop()


if __name__ == "__main__":
    main()
