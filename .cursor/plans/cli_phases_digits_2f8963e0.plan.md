---
name: CLI phases digits
overview: Add a `--phases` CLI flag (digit string like `0123` / `23`) that builds `phases_to_generate`, consolidating CLI parsing with the existing `--tbin` into one early argparse block.
todos:
  - id: hoist-cli-parser
    content: Move single argparse block (with --tbin and --phases) above PHASES TO GENERATE
    status: completed
  - id: wire-phases-digits
    content: Replace phases_to_generate comment cascade with digit-string decode + print
    status: completed
  - id: dedupe-tbin-parse
    content: Remove later duplicate parser; keep tbin application using early _cli_args
    status: completed
isProject: false
---

# Add `--phases` CLI digit encoding

## Encoding

```python
_PHASE_DIGIT_MAP = {
    '0': ProcessingScriptPhases.clean_run,
    '1': ProcessingScriptPhases.continued_run,
    '2': ProcessingScriptPhases.final_run,
    '3': ProcessingScriptPhases.figure_run,
}
```

`--phases 0123` → all four; `--phases 23` → final then figure. Digit **order in the string** is the phase list order. Unknown digits raise `ValueError`.

**Default when omitted: `3`** (matches the currently active `phases_to_generate = [ProcessingScriptPhases.figure_run]`).

## Changes in [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy)

1. **Hoist a single CLI parse** to just above the PHASES TO GENERATE section (~L213), so both flags are available early:
   - keep `--tbin` (default `'0.025'`)
   - add `--phases` (type `str`, default `'3'`)
   - still `parse_known_args()` for IPython compatibility

2. **Replace the comment-toggle cascade** (L216–223) with:

```python
# --phases digit codes: 0=clean_run, 1=continued_run, 2=final_run, 3=figure_run
# e.g. --phases 0123 (all), --phases 12 (continued+final), --phases 23 (final+figure); default 3
phases_to_generate = []
for _d in str(_cli_args.phases):
    if _d not in _PHASE_DIGIT_MAP:
        raise ValueError(f'Unknown --phases digit {_d!r} in {_cli_args.phases!r}; expected only 0/1/2/3')
    phases_to_generate.append(_PHASE_DIGIT_MAP[_d])
## END for _d in str(_cli_args.phases)...
print(f'cli --phases={_cli_args.phases!r} -> phases_to_generate={[p.value for p in phases_to_generate]}')
```

3. **Remove the later duplicate `_cli_parser` block** near the tbin assignment (~L319–325); keep only the tbin *application* there (`cli_tbin = parse_decoding_time_bin_size_str(_cli_args.tbin)` … suffix update), using `_cli_args` from the early parse.

## Invocation

```bash
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --tbin 0.050 --phases 23
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --phases 0123
```

## Scope

Only this `.ipy` file. No changes to `ProcessingScriptPhases` in the library.
