---
name: Path formatter buttons
overview: "Extend the Path IPython formatter with three action buttons: Copy (posix), Path(...) snippet, and Explorer (system file manager), using an ipywidgets row so Explorer can call reveal_in_system_file_manager."
todos:
  - id: link-html
    content: Simplify path_preview_repr_html to posix link-only HTML
    status: completed
  - id: path-widget
    content: Add path_preview_widget with Copy (posix), Path(...), Explorer buttons
    status: completed
  - id: registrar
    content: Update path_repr_with_filesystem_link to display the widget for Path types
    status: completed
isProject: false
---

# Path formatter: posix Copy, Path(...), Explorer

## Why widgets

Browser HTML can do clipboard; **Explorer** must call kernel Python via [`reveal_in_system_file_manager`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\Filesystem\open_in_system_file_manager.py) (same as [`fullwidth_path_widget`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\gui\Jupyter\simple_widgets.py)). Switch the registered formatter to `display()` a compact ipywidgets row.

## File

Only [`display_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\pho_jupyter_preview_widget\display_helpers.py).

## Behavior

| Control | Action |
|---------|--------|
| Link text | `path.as_posix()`; `href=path.as_uri()` |
| **Copy** | `copy_to_clipboard(path.as_posix())` |
| **Path(...)** | `copy_to_clipboard(f"Path({path.as_posix()!r})")` — always `Path`, never `WindowsPath` |
| **Explorer** | `reveal_in_system_file_manager(path)`; disabled if path does not exist |

## API changes

1. Keep `path_preview_repr_html(path)` as the **link-only** HTML helper (posix display text + `file://` href). Drop the HTML Copy button from it (buttons move to the widget).

2. Add `path_preview_widget(path, include_action_buttons: bool = True) -> widgets.Box`:
   - `widgets.HTML` for the link
   - three compact `widgets.Button`s (Copy / Path(...) / Explorer) using the same layout style as `fullwidth_path_widget`
   - wire clicks to `copy_to_clipboard` / `reveal_in_system_file_manager` (lazy imports inside the function)

3. Update `path_repr_with_filesystem_link(ip, include_action_buttons: bool = True)`:
   - Register Path / WindowsPath / PosixPath
   - Formatter: `display(path_preview_widget(p, include_action_buttons=...))` (same pattern as `dataframe_show_more_button`)
   - Replace old `include_copy_button` param name with `include_action_buttons`

## Example copied payloads

For `E:\Dropbox (Personal)\...\figure2_OUT.pdf`:

- Copy → `E:/Dropbox (Personal)/.../figure2_OUT.pdf`
- Path(...) → `Path('E:/Dropbox (Personal)/.../figure2_OUT.pdf')`
