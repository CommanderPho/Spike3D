---
name: All t_idx PF plot
overview: Add a new DisjointPlacefieldsExploration plotting method that shows static placefields/occupancy on the left and one posterior+spike-raster column per co-firing t_idx, reusing helpers extracted from the existing slider-based plot without changing its public API.
todos:
  - id: extract-helpers
    content: Extract _plot_single_tuning_map_2D, PF-stack builder, spike-row drawer as shared helpers; rewire plot_pfs_and_decoded_posterior to use them without API changes
    status: completed
  - id: add-all-t-method
    content: "Add plot_pfs_and_decoded_posteriors_all_t with 2-row GridSpec: left PF+occupancy, right columns for every t_idx posterior+spikes"
    status: completed
  - id: verify-both
    content: Smoke-check new all-t figure and confirm slider method still works unchanged
    status: completed
isProject: false
---

# All-t_idx Disjoint Placefields Plot

**Goal:** One static multi-column figure for a single cell pair showing all co-firing decoded posteriors side-by-side, with shared PF/occupancy on the left.

**Architecture:** Extract shared plotting helpers from [`plot_pfs_and_decoded_posterior`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) into private classmethods on `DisjointPlacefieldsExploration`; keep the existing slider method’s signature and behavior; add a sibling method that builds a 2-row GridSpec layout for all `t_idx`.

**Tech stack:** matplotlib `GridSpec`, existing `_plot_single_tuning_map_2D` / posterior / PF-stack / spike-raster logic already in that class.

## Layout

```text
Col0 (static)     Col1..N (one per t_idx)
[ Placefields ]   [ p(x|n) t_idx=0 ] ... [ p(x|n) t_idx=N-1 ]
[ Occupancy   ]   [ spikes t=0     ] ... [ spikes t=N-1     ]
```

- Left column always includes both placefields and occupancy (no `include_occupancy=False` for this variant).
- Each posterior axes title includes `t_idx` (and global bin when `co_firing_time_bin_indices` is provided).
- Figure width scales with `n_posterior_time_bins` (e.g. `~4.5 * (1 + n_t)`).

## API

New classmethod (name exact):

```python
@classmethod
def plot_pfs_and_decoded_posteriors_all_t(cls, neuron_sliced_decoder, co_firing_posteriors, tuple_key, nan_less_than_value: float = 1e-7, co_firing_time_bin_indices: Optional[NDArray] = None):
```

Caller selects the pair from compute outputs (same pattern as the notebook today):

```python
tuple_key = list(good_pairs_co_firing_posteriors_dict.keys())[2]
fig, axes = DisjointPlacefieldsExploration.plot_pfs_and_decoded_posteriors_all_t(
    neuron_sliced_decoder=good_pairs_co_firing_sliced_decoders_dict[tuple_key],
    co_firing_posteriors=good_pairs_co_firing_posteriors_dict[tuple_key],
    tuple_key=tuple_key,
    co_firing_time_bin_indices=good_pairs_co_firing_bins_dict[tuple_key],
)
```

Return `(fig, axes)` where `axes` is a structured array/dict of the left PF/occ axes plus per-`t_idx` posterior and spike axes (document in docstring). No slider.

## Reuse / refactor (do not break existing)

In [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) inside `DisjointPlacefieldsExploration`:

1. Lift nested `_plot_single_tuning_map_2D` to a private classmethod (or module-level private helper) used by both plot methods.
2. Extract helpers for:
   - building the RGBA placefield stack image + titles
   - drawing one spike-raster row for a given `t_idx`
   - applying `nan_less_than_value` to a posterior slice
3. Rewrite `plot_pfs_and_decoded_posterior` to call those helpers — **same public kwargs and slider behavior**.
4. Implement `plot_pfs_and_decoded_posteriors_all_t` using the same helpers; loop all `t_idx` in `range(n_posterior_time_bins)`.
5. Do **not** change `compute_and_plot_for_disjoint_cell_pairs` defaults (still uses the single-`t_idx`/slider plot when `plot_first_n_pairs` is set).

## Out of scope

- Notebook cell updates (caller can swap the loop for one call later).
- Design doc under `docs/superpowers/` (plan-mode; this plan is the approved artifact).
- Limiting / subsetting `t_idx` (always all).

## Verification

- Import and call the new method on one `tuple_key` from existing notebook dicts.
- Confirm existing `plot_pfs_and_decoded_posterior(..., t_idx=None)` still opens with slider and updates posterior + spikes only.
- Confirm left PF/occupancy appear once; right side has `n_bins` posterior+spike pairs labeled by `t_idx`.
