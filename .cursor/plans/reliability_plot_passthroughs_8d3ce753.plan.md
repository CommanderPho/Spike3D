---
name: Reliability plot passthroughs
overview: Add mixin passthroughs for the existing in-field mask plot, plus a new dual-panel (active|silent) reliability plot on CellIndividualReliabilityMatrix with matching mixin wrappers that pull state from the decoder.
todos:
  - id: passthru-masks
    content: Add CellIndividualReliabilityComputingMixin.plot_in_field_masks_with_spikes passthrough
    status: completed
  - id: plot-reliability
    content: Add CellIndividualReliabilityMatrix.plot_reliability_maps_with_spikes (active|silent dual panels, mode-aware)
    status: completed
  - id: passthru-reliability
    content: Add mixin plot_reliability_maps_with_spikes passthrough using self.reliability_* / mode / masks
    status: completed
isProject: false
---

# Reliability Plot Methods + Mixin Passthroughs

All changes in [`pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/reliability.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Analysis/reliability.py).

## 1. Mixin passthrough: `plot_in_field_masks_with_spikes`

Add an instance method on `CellIndividualReliabilityComputingMixin` that forwards to the existing classmethod:

```python
def plot_in_field_masks_with_spikes(self, included_neuron_ids=None, **kwargs):
    assert self.in_field_masks is not None
    return CellIndividualReliabilityMatrix.plot_in_field_masks_with_spikes(
        self.pf, self.in_field_masks,
        included_neuron_ids=included_neuron_ids if included_neuron_ids is not None else (self.neuron_IDs if self.neuron_IDs is not None else None),
        **kwargs,
    )
```

Place it in a new **Plotting Display** section at the end of the mixin (after `compute_reliability_metrics`).

## 2. New classmethod: `plot_reliability_maps_with_spikes`

Add next to `plot_in_field_masks_with_spikes` on `CellIndividualReliabilityMatrix`.

**Layout:** for each cell, two side-by-side axes — `reliability_active` | `reliability_silent`. Grid is `(n_cells, 2)` when `subplots` is None (one row per cell); if `subplots=(n_rows, n_cols)` is passed, treat it as the cell-grid and allocate `2 * n_cols` axes columns (active|silent per cell).

**Mode handling** (compare enums via `.value`):

- `POSITION_DEPENDENT`: `reliability_active` / `reliability_silent` are `(*spatial, n_neurons)` e.g. `(nx, ny, n_neurons)` or flat `(n_flat, n_neurons)`. Extract per-aclu spatial map via neuron index into `neuron_ids` order; reshape flat → `(nx, ny)` using `len(xbin)-1`, `len(ybin)-1`.
- `PER_CELL`: arrays are `(n_neurons,)`. Fill a constant `(nx, ny)` map with the scalar for that cell (so the same pcolormesh path works). Annotate title with the scalar value (e.g. `aclu 42  R_a=0.81  R_s=0.81`).

**Layers per panel (bottom → top):** optional trajectory → reliability heatmap (`vmin=0`, `vmax=1`, shared cmap e.g. `"viridis"`) → optional in-field mask overlay (if `in_field_masks` provided) → spikes (same in-field coloring as the existing plot when masks exist).

**Signature sketch** (single-line `def` per house style):

```python
@classmethod
def plot_reliability_maps_with_spikes(cls, pfs, reliability_active, reliability_silent, neuron_ids,
    reliability_estimation_mode: ReliabilityEstimationMode = ReliabilityEstimationMode.PER_CELL,
    in_field_masks: Optional[Dict[int, np.ndarray]] = None, included_neuron_ids=None, ...,
    which: str = "both",  # both | active | silent — default both
) -> Tuple[Figure, np.ndarray]:
```

Reuse subplot sizing / spike / trajectory / pcolormesh vs imshow patterns from `plot_in_field_masks_with_spikes`; do not rewrite that method — copy the shared scaffolding with minimal divergence.

**Asserts:** 2D `pfs`; `reliability_*` not None; lengths match `neuron_ids`; for POSITION_DEPENDENT, spatial size matches occupancy.

## 3. Mixin passthrough: `plot_reliability_maps_with_spikes`

```python
def plot_reliability_maps_with_spikes(self, included_neuron_ids=None, **kwargs):
    assert self.reliability_active is not None and self.reliability_silent is not None
    neuron_ids = np.asarray(self.neuron_IDs if self.neuron_IDs is not None else self.ratemap.neuron_ids)
    return CellIndividualReliabilityMatrix.plot_reliability_maps_with_spikes(
        self.pf, self.reliability_active, self.reliability_silent, neuron_ids,
        reliability_estimation_mode=getattr(self, 'reliability_estimation_mode', ReliabilityEstimationMode.PER_CELL),
        in_field_masks=self.in_field_masks, included_neuron_ids=included_neuron_ids, **kwargs,
    )
```

## Usage after change

```python
a_dst_decoder2D.plot_in_field_masks_with_spikes(max_n_cells=9)
a_dst_decoder2D.plot_reliability_maps_with_spikes(max_n_cells=9)
```
