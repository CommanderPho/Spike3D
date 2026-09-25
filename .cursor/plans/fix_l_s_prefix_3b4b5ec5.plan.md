---
name: Fix L/S prefix
overview: "Minimal fix: use pre/post-delta to choose `L` vs `S` when building multi-output bottom timestamp labels, instead of always hardcoding `L`."
todos:
  - id: fix-media-helper
    content: In _build_mergedColorDecoders_image_export_functions_dict, compute is_post_delta first and use L/S track_prefix
    status: completed
  - id: fix-data-exporting
    content: In data_exporting combined/multi label block, same L/S track_prefix fix
    status: completed
isProject: false
---

# Fix Long/Short prefix on multi timestamp labels

## Fault

Both label builders hardcode `L` before (or independently of) the existing `is_post_delta` branch that already sets blue/red text correctly.

## Change

In both sites, compute `is_post_delta` **before** building `complete_epoch_identifier_str`, then:

```python
track_prefix: str = 'S' if is_post_delta else 'L'
complete_epoch_identifier_str = f"{complete_epoch_identifier_str}{track_prefix}{active_epoch_id:03d}"
```

### 1. Shared helper (source of the pattern)

[`media_output_helpers.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers\plotting\media_output_helpers.py) — `ImagePostRenderFunctionSets._build_mergedColorDecoders_image_export_functions_dict`

- Move `is_post_delta: bool = (is_epoch_pre_post_delta[i] > 0)` above the identifier-string construction
- Replace hardcoded `L` at ~1022 with `track_prefix` as above

### 2. Multi combined export path (what the screenshot uses)

[`data_exporting.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\data_exporting.py) — combined/multi label block (~1582–1617)

- Move `is_post_delta: bool = (active_epoch_info_dict['pre_post_delta_category'] != 'pre-delta')` above the identifier-string construction
- Replace hardcoded `L` at ~1588 the same way

No other call sites, API changes, or notebook edits.