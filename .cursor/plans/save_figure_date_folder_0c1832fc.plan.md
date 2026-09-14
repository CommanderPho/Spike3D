---
name: Save figure date folder
overview: Add a configurable preferred output folder on DataFrameFilter (stored for later programmatic use) and by default date-prefix the Solara download basename as `YYYY-MM-DD_...png`.
todos:
  - id: add-folder-date-fields
    content: Add default_output_folder, include_date_prefix fields and date import
    status: completed
  - id: date-prefix-filename
    content: Update on_widget_update_filename to date-prefix basename and sync download/label
    status: completed
  - id: preferred-output-path
    content: Add preferred_output_path property joining folder + filename
    status: completed
isProject: false
---

# Save Figure date prefix + output folder

## Scope

Filename/label only (option 2): browser download basename gets a date prefix; a preferred output folder is stored on `DataFrameFilter` but Save Figure does **not** write to disk.

Only touch [`PhoDiba2023Paper.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PhoDiba2023Paper.py).

## Changes

### 1. New fields on `DataFrameFilter`

Near existing `filename` (~2749):

- `default_output_folder: Optional[Path] = non_serialized_field(default=None)` — settable at init or later (`df_filter.default_output_folder = Path(...)`)
- `include_date_prefix: bool = non_serialized_field(default=True)` — default on

Add `from datetime import date` at module imports (Path already imported).

### 2. Helper to build the download basename

Small method or inline in `on_widget_update_filename`:

- Base stem from `preferred_filename` or title (same as today, without `.png` until the end)
- If `include_date_prefix` and stem does not already start with `\d{4}-\d{2}-\d{2}_`, prefix with `f"{date.today().strftime('%Y-%m-%d')}_"`
- `self.filename = f"{stem}.png"`
- Sync `self.button_download.filename = self.filename` (already present)

### 3. Label + full path property

- `filename_label.value` shows the constructed basename (with date prefix), so it matches the download name
- Add a read-only `@property` `preferred_output_path` → `None` if folder unset, else `Path(self.default_output_folder) / self.filename` for later programmatic saves

### 4. Usage (no notebook edits)

```python
df_filter.default_output_folder = Path(r'E:/.../figures')
# Save Figure downloads e.g. 2026-09-14_preferred_name.png
# later: df_filter.preferred_output_path  # folder / that basename
```

## Not changing

- Solara lazy download / filename trait sync already in place
- No server-side PNG write on click
- No notebook / plotly_helpers changes
