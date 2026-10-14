---
name: Path link formatter
overview: Add an IPython text/html formatter in pho_jupyter_preview_widget that renders pathlib.Path (and concrete subclasses) as a clickable file:// link plus a Copy button, matching existing display_helpers registration patterns.
todos:
  - id: html-helper
    content: Add path_preview_repr_html() in display_helpers.py (file:// link + JS Copy button, safe escaping)
    status: completed
  - id: registrar
    content: Add path_repr_with_filesystem_link(ip) registering Path/WindowsPath/PosixPath for text/html
    status: completed
isProject: false
---

# Path filesystem-link IPython formatter

## Placement

Add two functions at the end of [`display_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\pho_jupyter_preview_widget\display_helpers.py), next to `array_repr_with_graphical_preview` / `dataframe_show_more_button`.

No new package, no magics, no `preferences_helpers` duplicate.

## API

```python
def path_preview_repr_html(path, include_copy_button: bool = True) -> str:
    """Self-contained HTML: clickable file link + optional Copy button."""

def path_repr_with_filesystem_link(ip, include_copy_button: bool = True) -> ip:
    """Register text/html formatters for Path / WindowsPath / PosixPath; return ip."""
```

Usage (same style as ndarray preview):

```python
from pyphocorehelpers.pho_jupyter_preview_widget.display_helpers import path_repr_with_filesystem_link
ip = get_ipython()
ip = path_repr_with_filesystem_link(ip=ip)
```

## Behavior

- Accept `pathlib.Path` (and subclasses); coerce `str` via `Path(...)` if needed for the HTML helper.
- Link: `href=path.as_uri()` (handles spaces / Windows), display text = `str(path)` (readable OS path, not `WindowsPath('...')`).
- Copy button: browser `navigator.clipboard.writeText(...)` of the same OS path string (matches [`fullwidth_path_widget`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\gui\Jupyter\simple_widgets.py)).
- Escape HTML attribute/text with `html.escape`; escape the JS string payload (backslashes, quotes, newlines) so Dropbox/space-heavy paths are safe.
- Compact inline row: path link + small `Copy` button; no ipywidgets (formatter must return an HTML `str`).
- Keep plain-text repr unchanged so Paths inside lists/dicts still print normally.

## Registration

IPython `for_type` is exact-type lookup, so register all three:

- `pathlib.Path`
- `pathlib.WindowsPath`
- `pathlib.PosixPath`

```python
html_formatter = ip.display_formatter.formatters['text/html']
for path_type in (Path, WindowsPath, PosixPath):
    html_formatter.for_type(path_type, lambda p, _include=include_copy_button: path_preview_repr_html(p, include_copy_button=_include))
return ip
```

Reuse link idea from `simple_path_display_widget` (`a_path.as_uri()`), but keep this formatter self-contained in `display_helpers.py` (no ipywidgets dependency on the path widgets).

## Docs in docstring

Include a short usage example showing before (`WindowsPath('E:/...')`) vs after (clickable link + Copy).
