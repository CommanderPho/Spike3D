---
name: clusterless-neuropy-object
overview: Move the portable sparse clusterless spike-event container into NeuroPy core while keeping pyPhoPlaceCellAnalysis as the Phy/RTC adapter layer. Preserve existing import paths and `.clusterless_spikes.npz` files through re-exports and compatibility aliases.
todos:
  - id: add-neuropy-class
    content: Add `neuropy.core.clusterless_spike_events.ClusterlessSpikeEvents` with NeuroPy-style validation, slicing, dict/dataframe, and NPZ persistence APIs.
    status: completed
  - id: export-neuropy-symbols
    content: Export the new class and helper functions from `neuropy.core.__init__`.
    status: completed
  - id: shim-pypho-adapter
    content: Replace the local pyPho dataclass/save/load definitions with imports from NeuroPy while keeping Phy and RTC adapter functions in pyPho.
    status: completed
  - id: add-neuropy-tests
    content: Add focused NeuroPy tests for construction, slicing, filtering, serialization, and NPZ compatibility.
    status: completed
  - id: verify-pypho-integration
    content: Run existing pyPho clusterless decoder tests to confirm old imports and RTC materialization still work.
    status: completed
isProject: false
---

# Clusterless Spike Events NeuroPy Object Plan

## Chosen Design

Create `ClusterlessSpikeEvents` as a NeuroPy core object in `[NeuroPy/neuropy/core/clusterless_spike_events.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/core/clusterless_spike_events.py)`. It should represent sparse clusterless spike events, not sorted units, so it will follow `FlattenedSpiketrains`/`Neurons` container conventions without inheriting unit-slicing behavior.

The class should inherit `StartStopTimesMixin`, `TimeSlicableObjectProtocol`, and `DataWriter`. Do not use `NeuronUnitSlicableObjectProtocol`, `SpikesAccessor`, or HDF mixins in this pass.

## Object API

Use `t_stop` as the canonical NeuroPy field, with a `t_end` property alias for existing pyPho code and saved-file compatibility.

Core fields:
- `spike_times_sec: np.ndarray`
- `electrode_indices: np.ndarray`
- `marks: np.ndarray`
- `sampling_frequency_hz: float = 1000.0`
- `electrode_mode: str = "channel"`
- `n_mark_dims: Optional[int] = None`, derived from `marks.shape[1]` when omitted
- `t_start: float = 0.0`
- `t_stop: Optional[float] = None`, defaulting to max spike time or `0.0` for empty events
- `source_phy_path: Optional[str] = None`
- `metadata: Optional[dict] = None`

Add focused NeuroPy-style conveniences:
- `__len__`, `n_spikes`, `n_electrodes`, `time`, `time_variable_name`
- `time_slice(t_start=None, t_stop=None)` that masks all parallel arrays consistently and returns a new `ClusterlessSpikeEvents`
- `get_by_electrode(electrode_indices)` for electrode-specific filtering without pretending electrodes are neurons
- `to_dataframe()` with columns `t_seconds`, `electrode`, and `mark_0...mark_N`
- `to_dict()` / `from_dict()` for DataWriter-style representation
- `save()` / `from_file()` or `to_npz()` / `from_npz()` backed by the existing versioned `.npz` schema

## Persistence And Compatibility

Move these symbols from `[pyPhoPlaceCellAnalysis/.../rtc_clusterless_adapters.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/rtc_clusterless_adapters.py)` to the new NeuroPy module:
- `CLUSTERLESS_SPIKE_EVENTS_FILE_VERSION`
- `ClusterlessSpikeEvents`
- `default_clusterless_spike_events_path`
- `save_clusterless_spike_events`
- `load_clusterless_spike_events`

Keep the `.npz` keys unchanged: `version`, `spike_times_sec`, `electrode_indices`, `marks`, `sampling_frequency_hz`, `electrode_mode`, `n_mark_dims`, `t_start`, `t_end`, `source_phy_path`. Internally load `t_end` into `t_stop`, but write `t_end` back out so existing tools and files remain compatible.

Export the class and helpers from `[NeuroPy/neuropy/core/__init__.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/neuropy/core/__init__.py)`.

## pyPho Adapter Changes

In `[rtc_clusterless_adapters.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/Decoder/rtc_clusterless_adapters.py)`, replace the local dataclass and save/load/default path definitions with imports from `neuropy.core.clusterless_spike_events`.

Keep these pyPho-specific functions in place:
- Phy/Kilosort extraction helpers
- `extract_clusterless_spike_events_from_phy_folder(...)`
- `build_multiunits_from_spike_events(...)`
- `build_multiunits_from_phy_folder(...)`
- RTC/PfND helpers and `ClusterlessDecodingParameters`

Update construction calls to pass `t_stop=` instead of `t_end=` where practical. Existing callers can continue to read `events.t_end` through the alias.

## Tests

Add `[NeuroPy/tests/test_clusterless_spike_events.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/NeuroPy/tests/test_clusterless_spike_events.py)` covering:
- constructor validation for aligned event arrays and mark dimensions
- `n_spikes`, `n_electrodes`, `time`, `t_end` alias
- `time_slice(...)` preserves aligned rows and updates `t_start`/`t_stop`
- `get_by_electrode(...)`
- `to_dataframe()` column shape and mark columns
- `to_dict()` / `from_dict()` round-trip
- `save_clusterless_spike_events(...)` / `load_clusterless_spike_events(...)` round-trip with existing `t_end` key
- unsupported version error

Keep pyPho RTC tests in `[pyPhoPlaceCellAnalysis/tests/test_rtc_clusterless_decoder.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/tests/test_rtc_clusterless_decoder.py)` mostly unchanged to verify the re-export and adapter integration path.

## Verification

Run focused tests with `uv` from the relevant repo roots:
- `uv run pytest tests/test_clusterless_spike_events.py -q` in `NeuroPy`
- `uv run pytest tests/test_rtc_clusterless_decoder.py -q` in `pyPhoPlaceCellAnalysis`

If imports fail because the editable multi-repo environment is not active, verify with the project’s existing environment activation path rather than adding package-management work to this change.