---
name: Fix PfND included_neuron_IDs
overview: Update `PfND.included_neuron_IDs` to return `np.asarray(self._ratemap.neuron_ids)` when `self._ratemap` is populated (avoiding recursion in `PfND_TimeDependent` and fixing `IndexError`/ordering bugs when `exact_neuron_ids` has zero-spike or custom-ordered neurons), and add unit test assertions.
todos:
  - id: fix-pfnd-included-neuron-ids
    content: Update PfND.included_neuron_IDs in NeuroPy/neuropy/analyses/placefields.py to return np.asarray(self._ratemap.neuron_ids) when self._ratemap is populated, falling back safely for PfND_TimeDependent
    status: completed
  - id: test-pfnd-included-neuron-ids
    content: Add regression assertions for PfND.included_neuron_IDs (0-spike sub-epoch, custom non-ascending order, NDArray return type) in pyPhoPlaceCellAnalysis/tests/test_decoders.py and run unittest
    status: completed
isProject: false
---

# Fix `PfND.included_neuron_IDs` for `exact_neuron_ids` and Zero-Spike Sub-Epochs

## Root Cause & Subclass Constraint

In [NeuroPy/neuropy/analyses/placefields.py](NeuroPy/neuropy/analyses/placefields.py) (`lines 1432-1435`), `PfND.included_neuron_IDs` currently indexes `self._filtered_spikes_df.spikes.neuron_ids` with `self.included_neuron_IDXs`:

```python
    @property
    def included_neuron_IDs(self):
        """The neuron IDs ('aclu' values) that were included after filtering by frate and etc. """
        return self._filtered_spikes_df.spikes.neuron_ids[self.included_neuron_IDXs] ## TODO: these are basically wrong, we should use self.ratemap.neuron_IDs instead!
```

This fails in three cases:
- **Zero-spike units under `exact_neuron_ids` (`preserve_source_neuron_IDs=True`):** `self._included_thresh_neurons_indx` has length `len(target_neuron_ids)`, while `self._filtered_spikes_df.spikes.neuron_ids` only contains units that fired `>= 1` spike in the new epochs, raising `IndexError`.
- **Non-ascending `exact_neuron_ids` order (e.g. `[2, 1]`):** `self._filtered_spikes_df.spikes.neuron_ids` is sorted ascending (`np.unique`), so indexing by `0..N-1` returns `[1, 2]` instead of `[2, 1]`.
- **Important subclass constraint (`PfND_TimeDependent`):** [NeuroPy/neuropy/analyses/time_dependent_placefields.py](NeuroPy/neuropy/analyses/time_dependent_placefields.py) (`lines 136, 254-262`) subclasses `PfND`, leaves `self._ratemap = None`, and overrides `@property def ratemap(self)` to pass `neuron_ids=self.included_neuron_IDs`. Checking `self._ratemap` (the backing field, **not** the `self.ratemap` property) avoids infinite recursion on `PfND_TimeDependent` while returning `np.asarray(self._ratemap.neuron_ids)` (`NDArray`) on computed/deserialized `PfND` instances.

## Planned Changes

### 1. Update `PfND.included_neuron_IDs` in `NeuroPy`
- File: [NeuroPy/neuropy/analyses/placefields.py](NeuroPy/neuropy/analyses/placefields.py) (`lines 1432-1436`)
- Update `included_neuron_IDs` (maintaining two blank lines between class methods):

```python
    @property
    def included_neuron_IDs(self):
        """The neuron IDs ('aclu' values) that were included after filtering by frate and etc. """
        if (self._ratemap is not None) and (getattr(self._ratemap, 'neuron_ids', None) is not None):
            return np.asarray(self._ratemap.neuron_ids)
        return self._filtered_spikes_df.spikes.neuron_ids[self.included_neuron_IDXs] ## fallback for pre-compute PfND or PfND_TimeDependent (where _ratemap is None)
```

### 2. Add Regression Assertions in `test_decoders.py`
- File: [pyPhoPlaceCellAnalysis/tests/test_decoders.py](pyPhoPlaceCellAnalysis/tests/test_decoders.py) (`lines 497-530`)
- Assert `list(pf_disjoint_preserved.included_neuron_IDs) == list(pf_initial.ratemap.neuron_ids)` on the 0-spike disjoint epoch (`[4.05, 4.85]`).
- Assert `list(train_dec.pf.included_neuron_IDs) == [2, 1]` and `list(silent_train_dec.pf.included_neuron_IDs) == [2, 1]` for both non-ascending custom order and 0-spike silent sub-epochs, plus `isinstance(..., np.ndarray)`.
- Run `TestGetByIdSlicesPfSpikesDfForEpochReplacement` via `uv run python -m unittest` to verify.