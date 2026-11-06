---
name: CLI tbin kwargs
overview: Add a `--tbin` CLI flag to `ProcessBatchOutputs_qclus1246789_Only.ipy` so the decoding time-bin size (and `_tbin_{N}ms` job suffix) are set from the command line instead of by uncommenting one of the hardcoded blocks.
todos:
  - id: import-parse-helper
    content: Import parse_decoding_time_bin_size_str alongside build_decoding_time_bin_size_filename_suffix
    status: completed
  - id: add-argparse-tbin
    content: Add argparse --tbin (default 0.025) with parse_known_args near the tbin config block
    status: completed
  - id: replace-comment-cascade
    content: Replace uncomment-toggle blocks with programmatic laps/ripple sizes + _tbin_{N}ms base_job_suffix
    status: completed
isProject: false
---

# Add `--tbin` CLI to ProcessBatchOutputs

## Goal

Replace the manual comment-toggle cascade in [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy) (lines ~318–344) with a single assignment driven by `--tbin`, e.g.:

```bash
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --tbin 0.025
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --tbin 0.050
```

The bare `--` is an **IPython argv separator**, not part of our flag: everything before it is for IPython; everything after is forwarded into `sys.argv` for the script. Without it, IPython may treat `--tbin` as its own unknown option and error. Plain `python script.ipy --tbin 0.050` needs no separator.

Same effect as uncommenting the 25ms or 50ms block (and works for 16/20/60/75ms too).

## Approach

In the Custom Suffix section, right where the tbin blocks live today:

1. **Parse CLI once** with `argparse` + `parse_known_args()` (so IPython’s own argv noise is ignored; works with `ipython script.ipy -- --tbin …`).
2. **`--tbin`**: float seconds, **default `0.025`** (matches the currently active block). Reuse [`parse_decoding_time_bin_size_str`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\NeuropyPipeline.py) so `--tbin 25ms` / `--tbin 0.025` both work; import it next to the existing `build_decoding_time_bin_size_filename_suffix` import.
3. **Replace the commented cascade** with:

```python
laps_decoding_time_bin_size = cli_tbin
ripple_decoding_time_bin_size = cli_tbin
if cli_tbin is not None:
    _tbin_ms = int(round(float(cli_tbin) * 1000.0))
    base_job_suffix = f"{base_job_suffix}_tbin_{_tbin_ms}ms"
```

4. Keep a short comment listing common values (`0.016`/`16ms`, `0.025`/`25ms`, `0.050`/`50ms`, …) so the old “menu” remains discoverable.
5. Leave the rest of the script unchanged — `decoding_time_bin_filename_suffix`, force-recompute kwargs, completion kwargs already consume `laps_decoding_time_bin_size` / `ripple_decoding_time_bin_size` / `base_job_suffix`.

## Invocation after change

```bash
# default still 25ms if omitted
ipython ProcessBatchOutputs_qclus1246789_Only.ipy

# IPython: separator `--` then script flags
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --tbin 0.050
ipython ProcessBatchOutputs_qclus1246789_Only.ipy -- --tbin 50ms

# plain python: no separator needed
python ProcessBatchOutputs_qclus1246789_Only.ipy --tbin 0.050
```

## Scope

Only [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy). No helper-library changes. No generic multi-kwarg framework beyond `--tbin` (easy to extend later with more `add_argument`s).
