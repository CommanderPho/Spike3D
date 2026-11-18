---
name: Generalize archive qclus flag
overview: Add a `--qclus` CLI flag to the existing archive script so it selects the correct `override_custom_pickle_suffix` for either qclus12 or qclus1246789, keeping sessions, move logic, and dry-run/execute behavior unchanged.
todos:
  - id: add-qclus-presets
    content: Replace PARAMETER_SPECIFIER with QCLUS_PARAMETER_SPECIFIERS map for qclus12 and qclus1246789
    status: completed
  - id: wire-cli-flag
    content: Add --qclus argparse flag (default qclus12) and use it in main + docstring
    status: completed
isProject: false
---

# Generalize archive script with `--qclus` flag

## Context

[`scripts/archive_qclus12_75ms_pickles.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\scripts\archive_qclus12_75ms_pickles.py) hardcodes one pickle suffix:

```python
PARAMETER_SPECIFIER = "_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0"
```

Active suffixes from the batch notebooks (same session list in both):

| Flag value | Source notebook | `parameter_specifier` |
|---|---|---|
| `qclus12` (default) | [`ProcessBatchOutputs_qclus12_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus12_Only.ipy) | `_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0` |
| `qclus1246789` | [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy) | `_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0` |

Move logic (`files_for_session`, `try_move`, `process_sessions`), `SESSION_REL_PATHS`, archive suffix, and dry-run/`--execute` stay as-is. Filename stays the same so existing docs keep working.

## Changes (single file)

In [`scripts/archive_qclus12_75ms_pickles.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\scripts\archive_qclus12_75ms_pickles.py):

1. Replace the lone `PARAMETER_SPECIFIER` constant with a preset map, e.g.:

```python
QCLUS_PARAMETER_SPECIFIERS = {
    "qclus12": "_withNormalComputedReplays-qclu_[1, 2]-frateThresh_2.0",
    "qclus1246789": "_withNormalComputedReplays-qclu_[1, 2, 4, 6, 7, 8, 9]-frateThresh_2.0",
}
```

2. Add `--qclus` to `parse_args` with `choices=list(QCLUS_PARAMETER_SPECIFIERS)`, `default="qclus12"`.

3. In `main`, resolve `parameter_specifier = QCLUS_PARAMETER_SPECIFIERS[args.qclus]` and print `qclus` + specifier in the header.

4. Update the module docstring usage block to show both modes, e.g.:

```bash
python archive_qclus12_75ms_pickles.py --qclus qclus12 --dry-run
python archive_qclus12_75ms_pickles.py --qclus qclus1246789 --execute
```

No notebook edits, no rename, no changes to session lists or archive-suffix behavior.