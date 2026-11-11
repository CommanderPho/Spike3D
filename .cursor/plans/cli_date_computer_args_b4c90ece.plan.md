---
name: CLI date computer args
overview: Extend the existing argparse in ProcessBatchOutputs_qclus1246789_Only.ipy so `--date` and `--computer` override BATCH_DAY_DATE / BATCH_DATE_TO_USE at script start, matching the current --tbin/--phases/--shuffles pattern.
todos:
  - id: move-cli-parser
    content: Move _cli_parser + parse_known_args above BATCH_DAY_DATE; add --date and --computer
    status: completed
  - id: wire-batch-date
    content: Set BATCH_DAY_DATE / BATCH_DATE_TO_USE from CLI; print resolved values; remove obsolete commented computer switch lines
    status: completed
isProject: false
---

# CLI `--date` / `--computer` for batch script

## Context

[`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy) hardcodes host+date here:

```75:80:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy
BATCH_DAY_DATE: str = '2026-11-12'
# BATCH_DATE_TO_USE = f'{BATCH_DAY_DATE}_Apogee' ...
BATCH_DATE_TO_USE = f'{BATCH_DAY_DATE}_GL' ...
_batch_host_suffix: str = ...
```

CLI already exists later for other knobs:

```226:230:h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument('--tbin', ...)
_cli_parser.add_argument('--phases', ...)
_cli_parser.add_argument('--shuffles', ...)
_cli_args, _ = _cli_parser.parse_known_args()
```

`BATCH_DATE_TO_USE` / `_batch_host_suffix` are used from line 80 onward, so date/computer must be resolved **before** those assignments (not at the current mid-file parser site).

## Approach

1. **Move** the existing `_cli_parser` + `parse_known_args()` block up to immediately above the `BATCH_DAY_DATE` assignment (after imports / timing helpers).
2. **Add** two args with defaults matching current hardcodes:
   - `--date` → default `'2026-11-12'` (current `BATCH_DAY_DATE`)
   - `--computer` → default `'GL'`; `choices=['Apogee', 'GL', 'Lab', 'rMBP']`
3. **Derive**:
   - `BATCH_DAY_DATE = _cli_args.date`
   - `BATCH_DATE_TO_USE = f'{BATCH_DAY_DATE}_{_cli_args.computer}'`
   - keep existing `_batch_host_suffix` logic unchanged
4. **Print** the resolved values once (same style as the existing `--phases` print) so CLI runs are auditable.
5. Leave the later phase/tbin usage of `_cli_args` as-is (same object, just parsed earlier). Downstream host-tag → `batch_execution_target` mapping stays driven by `BATCH_DATE_TO_USE` as today.

## Invocation

```bash
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --date 2026-11-12 --computer Apogee
```

(The `--` separator is required so IPython does not swallow script flags; same as existing `--tbin` / `--phases`.)

No docs / other-file changes unless you ask.