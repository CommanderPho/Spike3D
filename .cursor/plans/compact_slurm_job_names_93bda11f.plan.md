---
name: Compact SLURM job names
overview: Compact `#SBATCH --job-name` via `build_slurm_job_name` using your session/suffix rules, without changing pickle suffixes or long script filenames.
todos:
  - id: rewrite-builder
    content: Rewrite build_slurm_job_name with session+suffix compaction rules and IdentifyingContext support
    status: completed
  - id: update-callsite
    content: Pass IdentifyingContext object from _subfn_build_slurm_script instead of stringified context
    status: completed
  - id: annotate-ipy
    content: Update ProcessBatchOutputs_qclus1246789_Only.ipy comments to reference build_slurm_job_name
    status: completed
isProject: false
---

# Compact SLURM Job Names

**Goal:** Make SLURM job names short and distinguishable by applying your refinement rules in [`build_slurm_job_name`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py).

**Architecture:** Keep long `job_suffix` / pickle / script identifiers unchanged. Only compact the string written to `#SBATCH --job-name` in [`slurm_template.sh.j2`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Resources/Templates/slurm_template.sh.j2).

## Transformation (locked)

Example before (truncated at 64):
`job_kdiba_gor01_two_2006-6-12_16-53-46__withNormalComputedReplays-qcl`

After:
`gor01_2006-6-12_16-53-46_q1246789_fr_2_tbin_75ms`

Rules (from your `.ipy` notes):

**Session part**
- No `job_` prefix
- Drop `format_name` (`kdiba`)
- Drop `exper_name` (`one` / `two`)
- Keep `animal` + `session_name` → `gor01_2006-6-12_16-53-46`

**Suffix part**
- Drop `_withNormalComputedReplays` (default)
- `qclu_` → `q` (e.g. `qclu_1246789` → `q1246789`)
- `frateThresh_2.0` → `fr_2` (strip trailing `.0`; keep non-zero decimals like `fr_0.2`)
- Leave other tokens (e.g. `tbin_75ms`, `trackBodyPeakOnly`) as-is after cleanup

## Implementation

### 1. Rewrite `build_slurm_job_name` in [`pythonScriptTemplating.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py)

- Accept `curr_session_context` as `IdentifyingContext` or `str`.
- Short session string:
  - If context object: `get_description(subset_includelist=['animal', 'session_name'])`
  - If string: try `IdentifyingContext.try_init_from_session_key(...)`, then same; on failure, strip leading `kdiba_` only as fallback.
- Compact suffix with regex replacements:
  - remove `withNormalComputedReplays`
  - `qclu_` → `q`
  - `frateThresh_(\d+(?:\.\d+)?)` → `fr_<num>` with trailing `.0` stripped
  - collapse `_`/`-` runs; strip leading/trailing separators
- Join: `{short_session}_{compact_suffix}` (omit empty suffix)
- Sanitize commas/brackets/spaces (existing behavior), then `[:max_length]` (still 64)

### 2. Update call site in `_subfn_build_slurm_script`

Pass the context object (not `f"{a_curr_session_context}"`) so animal/session fields are available:

```python
slurm_job_name = build_slurm_job_name(curr_session_context=a_curr_session_context, job_suffix=job_suffix)
```

### 3. Point the `.ipy` notes at the implementation

In [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/Spike3D/ProcessBatchOutputs_qclus1246789_Only.ipy) (lines 226–236), keep the rules but note they are applied by `build_slurm_job_name` (so it does not look like unfinished TODO). Do not change the long `job_suffix` / pickle suffix values.

## Out of scope

- No change to `override_custom_pickle_suffix`, script filenames, or bash non-SLURM templates
- No new test file unless one already exists near this module (none found for `build_slurm_job_name`)
