---
name: Save Figure filename fix
overview: Fix Save Figure always downloading as `figure-image.png` by using a lazy Solara `FileDownload` data callable and updating the widget’s `filename` trait whenever the constructed name changes.
todos:
  - id: lazy-download-data
    content: Change _build_solera_file_download_widget to pass lazy pio.to_image callable as data
    status: completed
  - id: sync-filename-trait
    content: In on_widget_update_filename, set button_download.filename = self.filename; drop commented rebuild
    status: completed
isProject: false
---

# Fix Save Figure constructed filename

## Problem

[`_build_solera_file_download_widget`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) bakes PNG bytes and `filename="figure-image.png"` at construction. [`on_widget_update_filename`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py) updates `self.filename` / the label, but the rebuild of `button_download` is commented out, so the download control keeps the default name.

Verified: `solara.FileDownload.widget` exposes mutable `filename` and `data` traits (callable `data` is accepted).

## Approach (A, minimal)

Only touch [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py).

```mermaid
flowchart LR
  setup["_setup_widgets_buttons"] --> build["FileDownload.widget lazy data"]
  update["on_widget_update_filename"] --> setName["button_download.filename = self.filename"]
  click["Save Figure click"] --> lazy["callable runs pio.to_image"]
  click --> name["uses current filename trait"]
```

1. **`_build_solera_file_download_widget`** — stop eager `pio.to_image` / `deepcopy`. Pass a callable that captures `fig` and returns `pio.to_image(fig, format='png')` on click. Keep `filename` / `label` / `mime_type` args.

2. **`on_widget_update_filename`** — after setting `self.filename` and the label, sync the live widget:
   `self.button_download.filename = self.filename` (guard if button is None). Remove the commented rebuild assignment.

3. **Leave `_setup_widgets_buttons` creation as-is** — still builds with a placeholder name; the existing trailing `self.on_widget_update_filename()` call will set the constructed name immediately. No HBox child swap needed.

4. **Leave the duplicate commented rebuild** in `update_filtered_dataframes` (~4160) commented or delete the dead comment — that path already calls `on_widget_update_filename()`, so filename sync happens there.

## Not changing

- [`plotly_helpers.add_copy_save_action_buttons`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\plotly\Extensions\plotly_helpers.py) (separate JS download path)
- Notebook / UI layout / debounce logic
