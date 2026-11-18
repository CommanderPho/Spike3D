---
name: Archive date range
overview: Add optional `archive_start` / `archive_end` datetime bounds to `OldFileArchiver.archive_old_files`, filtering the input list by parsed filename `export_datetime` before moving into `OLD/`.
todos:
  - id: add-archive-range
    content: Add archive_start/archive_end to archive_old_files with parse_filename filter (inclusive start, exclusive end; skip unparsable when filtering)
    status: completed
isProject: false
---

# Add archive date range to `archive_old_files`

## Change

In [`AcrossSessionResults.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\AcrossSessionResults.py), extend `OldFileArchiver.archive_old_files`:

```python
@classmethod
def archive_old_files(cls, collected_outputs_directory: Path, excluded_or_outdated_files_list: List[Path], is_dry_run: bool=False, archive_start: Optional[datetime]=None, archive_end: Optional[datetime]=None):
```

## Filter semantics

- Both `None` (default): archive every path in the list (unchanged behavior).
- `archive_start` set: keep only files with `export_datetime >= archive_start`.
- `archive_end` set: keep only files with `export_datetime < archive_end` (half-open interval).
- Resolve `export_datetime` via existing same-module `parse_filename(path)` (first return value).
- If either bound is set and a path is unparsable (`export_datetime is None`): **skip** it (do not archive); print a short skip message.
- Assert `archive_start < archive_end` when both are provided.
- Print how many paths were kept vs skipped after filtering; then move only the filtered list.

Nest a tiny `_subfn_export_datetime_in_range(path) -> bool` inside the method (no new top-level helpers).

## Out of scope

- No changes to `cleanup_outdated_collected_outputs`, CLI, or notebooks (callers pass the new kwargs when needed).
- Docstring/usage example updated to show the new kwargs.