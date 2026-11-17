---
name: pylustrator matplotlib 3.8
overview: "Modernize pylustrator for Spike3D’s matplotlib==3.8.4: pin that version, remove pre-3.8 compatibility debt, fix known 3.8 breakages (toolbar `_active`, hard PyQt5 BoundSignal), then verify with offscreen pytest."
todos:
  - id: pin-mpl-384
    content: Pin matplotlib==3.8.4 in pyproject.toml; uv lock/sync; align __version__ to 1.3.2
    status: completed
  - id: fix-toolbar-active
    content: Replace navi_toolbar._active usage in plot_layout.py with public mode / prev_active tracking
    status: completed
  - id: fix-pyqt-boundsignal
    content: Remove hard PyQt5 pyqtBoundSignal imports; use Signal annotations in QLinkableWidgets and qitem_properties
    status: completed
  - id: drop-pre38-compat
    content: Remove axes._subplots, ncol, _dpi_ratio, backend_qt5agg fallbacks; harden _make_key add path
    status: completed
  - id: verify-tests
    content: Import smoke test + QT_QPA_PLATFORM=offscreen pytest (+ ty check); update CLAUDE.md
    status: completed
isProject: false
---

# Modernize pylustrator for matplotlib 3.8.4

## Scope

Package only: [`pylustrator`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator). No Spike3D path-dep wiring in this pass.

## Pin and lock

In [`pyproject.toml`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pyproject.toml), change:

```toml
"matplotlib>=2.0.2",
```

to:

```toml
"matplotlib==3.8.4",
```

Then regenerate the lock with `uv lock` / `uv sync --all-extras` so CI (`uv sync --locked`) resolves **3.8.4** instead of the current 3.9.4 / 3.10.8 entries.

Align version strings: [`pylustrator/__init__.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\__init__.py) still has `__version__ = "1.3.0"` while pyproject is `1.3.2` — set `__version__` to `1.3.2`.

## Fix known matplotlib 3.8 breakages

### 1. Toolbar `_active` (hard break since mpl 3.3+)

[`plot_layout.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\components\plot_layout.py) assigns/reads `self.navi_toolbar._active` (`"DRAG"` / `"PAN"` / `"ZOOM"`). That attribute does not exist on 3.8 and raises `AttributeError`.

Replace with public mode checks via `toolbar.mode` (string/`_Mode`: `""`, `"pan/zoom"`, `"zoom rect"`) and keep using `navi_toolbar.pan()` / `.zoom()` toggles. Track pylustrator’s custom DRAG state only in `prev_active` / action checks — do not invent a fake `_active = "DRAG"` on the mpl toolbar.

### 2. Hard `PyQt5.QtCore.pyqtBoundSignal`

Unconditional imports in [`QLinkableWidgets.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\QLinkableWidgets.py) and [`qitem_properties.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\components\qitem_properties.py) fight the existing `qtpy` runtime path. With `from __future__ import annotations` already present, drop those imports and annotate with `Signal` (already imported from qtpy / TYPE_CHECKING PyQt5).

### 3. `_make_key` pickle/reload path

In [`helper_functions.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\pylustrator\helper_functions.py) (~line 367), `figure._axstack.add(figure._make_key(ax), ax)` is private and fragile. Mirror the existing dual-path pattern used at lines 163–166: try `(key, ax)` / catch `TypeError` then `add(ax)`.

## Drop dead pre-3.8 branches (safe under exact pin)

Because the package will require 3.8.4 exactly:

| Location | Change |
|----------|--------|
| `snap.py`, `helper_functions.py`, `ax_rasterisation.py`, `qpos_and_size.py` | Keep only `from matplotlib.axes import Axes`; remove `axes._subplots` fallback |
| `change_tracker.py`, `qitem_properties.py` | Always use `ncols` (drop `<3.6` → `ncol`) |
| `plot_layout.py` `icon()` | Always use `canvas.device_pixel_ratio` (drop `_dpi_ratio` try) |
| Backend imports (`matplotlibwidget.py`, `plot_layout.py`, `QtGui.py`) | Prefer `backend_qtagg` only (3.8 has it); remove `backend_qt5agg` fallback |

Leave still-useful private APIs that work on 3.8 (`_pylab_helpers.Gcf`, `_axstack` with the dual `add` path where needed, legend `_loc` / `_fontsize`) unless tests prove they fail — no speculative rewrite of the whole TargetWrapper/ChangeTracker stack.

## Docs / notes

Update the dependency line in [`CLAUDE.md`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pylustrator\CLAUDE.md) from “Matplotlib 2.0+” to pinned **3.8.4**.

## Verification

```bash
uv sync --all-extras
uv run python -c "import matplotlib; assert matplotlib.__version__ == '3.8.4'; import pylustrator; from pylustrator.drag_helper import DragManager"
QT_QPA_PLATFORM=offscreen uv run pytest
uv run ty check   # if ty extra/dev present
```

Existing coverage (`tests/test_axes.py`, `test_text.py`, `test_legend.py`) exercises the GUI path with `hide_window=True` and is enough to catch toolbar init / ncols / Axes import regressions.

## Out of scope

- Spike3D `[tool.uv.sources]` path dependency
- Restoring missing `pylustrator/icons/` assets (referenced but empty in this checkout)
- Rewriting all private mpl artist access (`legend_`, tick `_gridOnMajor`, etc.)
- Bumping package major version beyond syncing `__version__` to 1.3.2
