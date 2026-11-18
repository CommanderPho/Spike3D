---
name: Filter PBE contact sheets
overview: Add an optional `example_PBE_ids_dict` parameter to `build_contact_sheets_for_session_contexts` so each session can export only a curated list of PBE images (matched via the `[N]` index in filenames like `p_x_given_n[8].png`), preserving the caller's ID order.
todos:
  - id: add-param
    content: Add example_PBE_ids_dict param + docstring usage to build_contact_sheets_for_session_contexts
    status: completed
  - id: filter-loop
    content: Filter images_dict by [N] index in caller order; honor skip_missing for missing IDs/empty results
    status: completed
isProject: false
---

# Filter contact sheets by example PBE IDs

## Target

[`pyphocorehelpers/src/pyphocorehelpers/plotting/media_output_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\plotting\media_output_helpers.py) — `build_contact_sheets_for_session_contexts` (lines ~1667–1739).

## Behavior

Add optional kwarg:

```python
example_PBE_ids_dict: Optional[Dict[IdentifyingContext, List[int]]] = None
```

- `None` (default): current behavior — all images in the multi folder.
- Context present in the dict: keep only images whose `\[(\d+)\]` index is in that session’s ID list, **in the order given** (not re-sorted).
- Context absent from the dict: still export all images for that session (partial dicts OK).
- Missing IDs / empty result after filter: reuse existing `skip_missing` (print + continue vs raise).

Image keys already come from [`ImageHelpers.load_png_images_pathlib`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\image_helpers.py) as stems like `p_x_given_n[8]`; index extraction mirrors that helper’s `extract_index`.

`IdentifyingContext` already implements value-based `__eq__`/`__hash__`, so dict lookup against separately constructed contexts works.

## Implementation (minimal)

Inside the per-context loop, after `images_dict = ImageHelpers.load_png_images_pathlib(...)`:

1. If `example_PBE_ids_dict` is set and `a_ctxt` is in it, build `idx -> (stem, img)` from filenames, then select in caller order.
2. Set `imgs` / `filenames` from the filtered list (or all images when no filter).
3. If filter yields zero images, treat like the existing empty-folder path via `skip_missing`.
4. When filtering, debug print e.g. `28/28 requested` (or which IDs were missing).
5. Update the docstring usage example to pass `example_PBE_ids_dict=active_example_PBE_ids_dict`.

No change to output filename pattern (`{strip_date_prefix}_Strip-from-multi_...`); callers can still override via `strip_date_prefix` / `strip_suffix`. No notebook edits unless you ask.

## Example call

```python
out_paths, out_images = build_contact_sheets_for_session_contexts(
    included_session_contexts,
    posteriors_root=src_dir,
    output_dir=out_path,
    combined_subdir='combined/multi',
    example_PBE_ids_dict=active_example_PBE_ids_dict,
    spacing=(2, 0), fill_mode=True,
)
```
