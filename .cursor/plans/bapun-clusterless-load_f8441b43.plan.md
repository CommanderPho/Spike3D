---
name: bapun-clusterless-load
overview: Enable Bapun session loading to opportunistically attach a saved clusterless spike-event NPZ to `session.clusterless_spike_events`, matching the notebook save path convention.
todos:
  - id: add-loader-helper
    content: Add Bapun helper to locate and load `{session_name}.clusterless_spikes.npz` into `session.clusterless_spike_events`.
    status: completed
  - id: wire-spike-comps
    content: Call the helper from `_perform_spike_comps` without changing existing flattened-spiketrain fallback behavior.
    status: completed
  - id: add-focused-test
    content: Add a synthetic NPZ roundtrip test for the Bapun helper and `session.clusterless_spike_events` assignment.
    status: completed
  - id: verify
    content: Run the narrow NeuroPy pytest target with `uv run pytest tests/test_clusterless_spike_events.py -q` and fix any introduced lints or test failures.
    status: completed
isProject: false
---

# Bapun Clusterless Load Plan

## Scope
Add an optional clusterless-event load to `[NeuroPy/neuropy/core/session/Formats/Specific/BapunDataSessionFormat.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/core/session/Formats/Specific/BapunDataSessionFormat.py)`. The existing required-file loading and `.flattened.spikes.npy` fallback computation will remain unchanged.

Relevant existing load hook:

```1061:1069:h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/core/session/Formats/Specific/BapunDataSessionFormat.py
        def _perform_spike_comps(session):
            ## Load or compute flattened spikes since this format of data has the spikes ordered only by cell_id:
            ## flattened.spikes:
            active_file_suffix = '.flattened.spikes.npy'
            # active_file_suffix = '.new.flattened.spikes.npy'
            found_datafile = FlattenedSpiketrains.from_file(session.filePrefix.with_suffix(active_file_suffix))
            if found_datafile is not None:
                print('Loading success: {}.'.format(active_file_suffix))
                session.flattened_spiketrains = found_datafile
```

## Approach
- Import `ClusterlessSpikeEvents`, `default_clusterless_spike_events_path`, and `load_clusterless_spike_events` from NeuroPy core near the existing `FlattenedSpiketrains` import.
- Add a small Bapun class helper, likely `_try_load_clusterless_spike_events_file(session)`, so the path and assignment can be unit-tested without constructing a full Bapun data directory.
- Resolve the filename as `basedir / f"{sess.name}.clusterless_spikes.npz"` by using `session.config.session_name` when available, falling back to `session.filePrefix.name`; use `session.filePrefix.parent` as `basedir`.
- If the NPZ exists, load it with `events: ClusterlessSpikeEvents = load_clusterless_spike_events(clusterless_save_path)`, set `events.filename = clusterless_save_path`, and assign `session.clusterless_spike_events = events`.
- If the file is absent, leave the session otherwise unchanged and do not recompute or create clusterless data.
- Call the helper from `_perform_spike_comps(session)` after the existing flattened-spiketrain load/compute path, so sorted spikes and flattened spikes keep their current behavior.

## Testing
Add a focused test in `[NeuroPy/tests/test_clusterless_spike_events.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/tests/test_clusterless_spike_events.py)` or a new small Bapun-format test file. The test will create a temporary file named `Synthetic.clusterless_spikes.npz`, save a tiny `ClusterlessSpikeEvents`, build a `SimpleNamespace` session with `filePrefix=tmp_path / "Synthetic"` and `config.session_name="Synthetic"`, call the new helper, and assert:
- `session.clusterless_spike_events` exists and is a `ClusterlessSpikeEvents`.
- Loaded spike times, electrode indices, and marks match the saved object.
- `session.clusterless_spike_events.filename` equals the NPZ path.

Run the narrow test command from `[NeuroPy](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy)`:

```shell
uv run pytest tests/test_clusterless_spike_events.py -q
```

If imports from `BapunDataSessionFormat.py` pull in optional plotting dependencies during test collection, use the same lightweight compatibility pattern already present in `tests/test_nwb_data_session_format.py` before importing the Bapun class.