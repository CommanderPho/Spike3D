---
name: Stage tiny sidecars
overview: Stage tiny Turbo session-root sidecars (MATLAB/xml/npy) as real files into each scratch session folder so FINAL post-load code can re-read them next to the pickle. Only edit ProcessBatchOutputs_qclus1246789_Only.ipy.
todos:
  - id: add-sidecar-list
    content: Add readonly sidecar suffix/name tuples in ProcessBatchOutputs_qclus1246789_Only.ipy
    status: completed
  - id: copy-sidecars-in-stage
    content: Copy those sidecars in stage_mutable_session_outputs_to_scratch with dest-mtime-newer skip; warn if position_info.mat is missing
    status: completed
isProject: false
---

# Stage tiny R/O sidecars to scratch

The pickle already loaded. [`PostHocPipelineFixup.FINAL_FIX_GRID_BIN_BOUNDS`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Batch\BatchJobCompletion\UserCompletionHelpers\batch_user_completion_helpers.py) then re-reads `{session}.position_info.mat` from `sess.basepath` (scratch after filepath remap). That file was never staged. Copy it and the other tiny session-root files you selected.

Leave [`KDibaOldDataSessionFormat.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\Specific\KDibaOldDataSessionFormat.py) and the FINAL helpers unchanged. Copy is simpler than symlink and reuses the existing dest-mtime-newer skip.

## Sidecars to copy (from Turbo session dir)

Named `{session_name}{suffix}`:

- `.position_info.mat` (the crash)
- `.epochs_info.mat`, `.xml`
- `.laps_info.mat`, `.replay_info.mat`, `.RippleDatabase.mat`
- `.probegroup.npy`, `.pbe.npy`, `.ripple.npy`, `.mua.npy`, `.position.npy`

Also copy standalone `RippleDatabase.mat` if present (folder listing uses that name; loader uses `{session}.RippleDatabase.mat`).

Do **not** copy: `.dat`/`.eeg`/`.clu`/`.fet`, `spikes.mat`/`spikeII.mat`, `.h5`, unused pickles.

Missing files: print and continue. Missing `{session}.position_info.mat` gets an extra `WARN` (FINAL will still crash without it).

## Change (driver only)

In [`ProcessBatchOutputs_qclus1246789_Only.ipy`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\Spike3D\ProcessBatchOutputs_qclus1246789_Only.ipy):

1. Add suffix/name tuples next to the staging helpers.
2. In `stage_mutable_session_outputs_to_scratch`, after the local/global pickle copies, copy each sidecar with `copy_or_skip_mutable_file` (`skip_if_dest_newer_or_equal` unchanged: dest mtime newer never overwrites, and still prints the WARN).
3. Update the function docstring. No call-site change.

Dest-mtime-newer skip is useful for `.pbe.npy` / `.ripple.npy` / `.mua.npy` / `.position.npy` if a later job wrote newer scratch copies.
