---
title: Merging 1D decoders into a pseudo-2D context decoder
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/8ceea2d3-45ef-40c5-bd94-dcbfa14e75aa
url: https://phohale.sourcegraph.app/deepsearch/381
created: 2026-07-17T02:47:48Z
updated: 2026-07-22T18:17:46Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/diba-lab/NeuroPy find my code that merges several 1D decoders into a "Pseudo2D" context decoder and renormalizes 

Now I have a complete picture. Here is a full summary of all the relevant code:

---

## Pseudo2D Context Decoder - How It Works

The code is spread across two key locations:

### 1. `PfND.build_merged_directional_placefields` — the core merge + renormalization

Defined in [`NeuroPy/neuropy/analyses/placefields.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1673-1867) (in the `Spike3DWorkEnv` monorepo copy, which is the authoritative source for the NeuroPy version you use):

**What it does:**

1. **Finds the union of all neuron IDs** across all 1D decoders (so cells active in any decoder are included, with zeros for those absent in a given decoder).
2. **Stacks tuning curves / spikes maps / occupancy** along a new pseudo-y axis via `np.stack(..., axis=-1)` — one slice per decoder (e.g. `long_LR`, `long_RL`, `short_LR`, `short_RL`). Each decoder occupies a distinct integer y-bin: decoder 0 → y=1, decoder 1 → y=2, etc. ([line 1832–1833](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1832-1833))
3. **Assigns a synthetic `ybin`** of `np.arange(n_decoders + 1)` (bin edges), so it looks like a 2D field even though the real position is 1D. ([line 1806](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1806))
4. **Sets config appropriately**: `grid_bin[pseudo_y] = 1.0`, `smooth[pseudo_y] = 0.0` (no smoothing across decoders), `is_directional = True`. ([lines 1854–1858](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1854-1858))
5. **Returns a `PfND` with `ndim = original_ndim + 1`** wrapping the new stacked `Ratemap`. The resulting object is the Pseudo2D decoder. The renormalization is handled downstream by the Bayesian decoder (`BasePositionDecoder`) when it calls its own `pdf_normalized_tuning_curves` during decoding.

> **Note:** The function explicitly clears each sub-decoder's `frate_thresh` to 0.0 and calls `.compute()` before merging to ensure consistent cell populations ([lines 1709–1710](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1709-1710)).

---

### 2. `_build_merged_directional_placefields` — the pipeline computation function

Defined in [`DirectionalPlacefieldGlobalComputationFunctions.py`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8103-8243), this is the global computation that orchestrates the merge:

- Extracts the four directional 1D placefields: `long_LR_pf1D`, `long_RL_pf1D`, `short_LR_pf1D`, `short_RL_pf1D` ([line 8153](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8153))
- Filters shared aclus by `minimum_inclusion_fr_Hz` and `qclu` ([line 8162](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8162))
- Calls `PfND.build_merged_directional_placefields(all_directional_pf_dict)` to produce the **all-4-decoder** Pseudo2D object ([line 8165](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8165))
- Also builds **long-only** (`long_LR` + `long_RL`) and **short-only** merged sub-decoders the same way ([lines 8173, 8180](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8173-8180))
- Stores everything in a [`DirectionalPseudo2DDecodersResult`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L2468) under `global_computation_results.computed_data['DirectionalMergedDecoders']`
- Decodes laps and ripples with the merged Pseudo2D decoder, then calls `perform_compute_marginals()` ([line 8227](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8227))

---

### 3. `DirectionalPseudo2DDecodersResult` — the result container

Defined at [line 2468](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L2468). Holds:
- `all_directional_pf1D_Decoder` — the 4-decoder Pseudo2D `BasePositionDecoder`
- `all_directional_decoder_dict` — the original four individual `PfND` objects
- `all_directional_laps_filter_epochs_decoder_result` / `all_directional_ripple_filter_epochs_decoder_result` — the posterior results decoded using the Pseudo2D decoder
- Methods like `perform_compute_marginals()`, `determine_directional_likelihoods()`, `determine_long_short_likelihoods()` for marginalizing over the pseudo-y axis after decoding

## Sources

- [repo:CommanderPho/pyPhoPlaceCellAnalysis Pseudo2D merge 1D decoder renormalize](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20Pseudo2D%20merge%201D%20decoder%20renormalize)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis Pseudo2D decoder](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20Pseudo2D%20decoder)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis build_merged_directional_placefields pseudo2D](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20build_merged_directional_placefields%20pseudo2D)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis renormalize pseudo2D merge 1D decoders](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20renormalize%20pseudo2D%20merge%201D%20decoders)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis build_merged_directional_placefields def](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20build_merged_directional_placefields%20def)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis _build_merged_directional_placefields def](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20_build_merged_directional_placefields%20def)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@0676630?L8100-8250](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8100-8250)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis def build_merged_directional_placefields](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20def%20build_merged_directional_placefields)
- [repo:diba-lab/NeuroPy merge directional placefields pseudo2D renormalize](/search?q=patterntype:nls%20repo:diba-lab%2FNeuroPy%20merge%20directional%20placefields%20pseudo2D%20renormalize)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis PfND build_merged_directional_placefields classmethod](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20PfND%20build_merged_directional_placefields%20classmethod)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis PfND build_merged_directional_placefields stacking concatenate renormalize](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20PfND%20build_merged_directional_placefields%20stacking%20concatenate%20renormalize)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis "build_merged_directional_placefields" file:.py](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22build_merged_directional_placefields%22%20file:.py)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis "PfND.build_merged" file:DirectionalPlacefieldGlobalComputationFunctions.py](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22PfND.build_merged%22%20file:DirectionalPlacefieldGlobalComputationFunctions.py)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis PfND mixin directional merged placefields classmethod definition](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20PfND%20mixin%20directional%20merged%20placefields%20classmethod%20definition)
- [github.com/diba-lab/NeuroPy](/github.com/diba-lab/NeuroPy)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis merge directional placefields stacking y-axis pseudo renormalize occupancy](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20merge%20directional%20placefields%20stacking%20y-axis%20pseudo%20renormalize%20occupancy)
- [build_merged_directional_placefields.*def|def.*build_merged_directional_placefields repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ patterntype:regex](/search?q=build_merged_directional_placefields.%2Adef%7Cdef.%2Abuild_merged_directional_placefields%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20patterntype:regex)
- [build_merged_directional_placefields repo:^github.com/diba-lab/NeuroPy$ patterntype:keyword](/search?q=build_merged_directional_placefields%20repo:%5Egithub.com%2Fdiba-lab%2FNeuroPy$%20patterntype:keyword)
- [def build_merged_directional_placefields repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ patterntype:regex](/search?q=def%20build_merged_directional_placefields%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20patterntype:regex)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis "PfND.build_merged_directional_placefields" = ](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22PfND.build_merged_directional_placefields%22%20=%20)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis file:DirectionalPlacefieldGlobalComputationFunctions.py build_merged_directional_placefields classmethod staticmethod](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20file:DirectionalPlacefieldGlobalComputationFunctions.py%20build_merged_directional_placefields%20classmethod%20staticmethod)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@0676630?L1-80](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L1-80)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@0676630?L81-200](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L81-200)
- [def build_merged_directional_placefields repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ patterntype:regex](/search?q=def%20build_merged_directional_placefields%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20patterntype:regex)
- [build_merged_directional_placefields patterntype:regex](/search?q=build_merged_directional_placefields%20patterntype:regex)
- [github.com/CommanderPho/Spike3DWorkEnv/NeuroPy/neuropy/analyses/placefields.py@eadd004?L1670-1770](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1670-1770)
- [github.com/CommanderPho/Spike3DWorkEnv/NeuroPy/neuropy/analyses/placefields.py@eadd004?L1770-1900](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1770-1900)


---

# How is the context decoder computed in the case of 2D positions (e.g. "Pseudo3D") -- specifically for the NWB Session format? Provide both the code process outline (in terms of code functions) and the mathematical/conceptual outline of the process of combination. 

Now I have the complete picture. Let me compile the full answer.

---

## Pseudo3D Context Decoder for 2D Positions (NWB/Bapun Sessions)

### Code Process Outline

The pipeline for a non-kdiba/NWB session diverges inside [`_decode_continuous_using_directional_decoders`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8307-8431). The key branching check is `is_kdiba_session`:

```
is_kdiba_session == False (NWB/Bapun)
  → build_contextual_pf2D_decoder(...)          [PendingNotebookCode.py:5903]
  → decode_using_contextual_pf2D_decoder(...)   [PendingNotebookCode.py:5944]
  → pack into DirectionalDecodersContinuouslyDecodedResult
```

---

#### Step 1 — Retrieve per-context 2D decoders

[`build_contextual_pf2D_decoder`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5903-5938) collects one `pf2D_Decoder` per named session epoch (e.g. `['roam', 'sprinkle']` for Bapun, or whatever `hardcoded_params.non_global_activity_session_names` resolves to for other NWB formats):

```python
pf2D_Decoder_dict = {k: deepcopy(curr_active_pipeline.computation_results[k].computed_data.pf2D_Decoder)
                     for k in epochs_to_create_global_from_names}
```

#### Step 2 — Align spatial bins across contexts

Before merging, each `pf2D` is conforming to a shared spatial grid via `.conform_to_position_bins(a_pf, force_recompute=True)` ([line 5929](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5925-5930)), so all contexts have the same `xbin` and `ybin`.

#### Step 3 — Merge via `PfND.build_merged_directional_placefields`

The same merge function used for Pseudo2D is called on 2D input placefields ([line 5932](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5932-5933)):

```python
contextual_pf2D: PfND = PfND.build_merged_directional_placefields(contextual_pf2D_dict, debug_print=False)
contextual_pf2D_Decoder: BasePositionDecoder = BasePositionDecoder(contextual_pf2D, ...)
```

Inside [`build_merged_directional_placefields`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1810-1817), the `ndim > 1` branch is taken:

```python
elif ((ndim > 1) and (ybin is not None)):
    zbin = np.arange(new_pseudo_num_VNEWbins + 1)  # [0, 1, ..., n_contexts]
    bin_kwargs.update(ybin=deepcopy(ybin), zbin=deepcopy(zbin))
```

This creates a **Pseudo3D** `PfND` with `ndim = 3` (x, y, context-z).

The stacking of tuning curves is identical to the 1D case, just applied to 2D arrays ([line 1832](https://phohale.sourcegraph.app/r/github.com/CommanderPho/Spike3DWorkEnv/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1832-1833)):

```python
stacked_results_dict = {
    key: np.stack([v[key] for v in at_least_one_decoder_all_results_dict.values()], axis=-1)
    for key in ['tuning_curves', 'unsmoothed_tuning_maps', 'spikes_maps']
}
stacked_occupancy = np.stack([v.occupancy for v in directional_1D_decoder_dict.values()], axis=-1)
```

For 2D inputs with shape `(n_neurons, n_xbins, n_ybins)`, the stacked tuning curves become `(n_neurons, n_xbins, n_ybins, n_contexts)`. Config is patched identically: `grid_bin[-1] = 1.0`, `smooth[-1] = 0.0`, no smoothing along z.

#### Step 4 — Decode

[`decode_using_contextual_pf2D_decoder`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5944-6017):

- Creates a merged global epoch spanning all named context epochs via `epochs_df.epochs.adding_concatenated_epoch(...)` ([line 5973](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5973))
- Runs `contextual_pf2D_Decoder.decode_specific_epochs(...)` at the desired time bin size ([line 5983](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5983))

The resulting posterior `p_x_given_n` has shape `(n_xbins, n_ybins, n_contexts, n_t_bins)`.

#### Step 5 — Compute context marginal (`marginal_z`)

After decoding, the context (z-axis) marginal is computed ([lines 6001–6003](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L6001-6003)):

```python
n_x_bins, n_y_bins, n_contexts, n_t_bins = np.shape(p_x_given_n)
marginal_z = np.nansum(p_x_given_n, axis=(0, 1))          # → (n_contexts, n_t_bins)
marginal_z = marginal_z / np.sum(marginal_z, axis=0, keepdims=True)  # normalize per time bin
```

This is stored as `all_context_filter_epochs_decoder_result.marginal_z`.

---

### Mathematical / Conceptual Outline

#### What is being computed

Each session epoch (e.g. "roam", "sprinkle") provides an independent 2D placefield — a probability distribution over 2D space for each neuron. The goal is to simultaneously decode **where** the animal is in 2D space **and which context** is most active, using all neurons jointly.

#### Tensor structure of the merged decoder

For $N$ neurons, $n_x \times n_y$ spatial bins, and $K$ contexts:

- Each per-context 2D tuning curve: $f_k^{(i)}(x, y)$ for neuron $i$, context $k$ — shape $(N, n_x, n_y)$
- The merged Pseudo3D tuning map stacks these along a new z-axis:

$$F^{(i)}(x, y, k) = f_k^{(i)}(x, y) \quad \in \mathbb{R}^{N \times n_x \times n_y \times K}$$

The resulting `PfND` has `ndim = 3` with spatial axes $(x, y)$ and a discrete pseudo-z axis over $k \in \{1, \ldots, K\}$.

#### Bayesian decoding of the joint posterior

At each time bin $t$ with observed spike counts $\mathbf{n}_t = (n_1, \ldots, n_N)$, the decoder applies the standard Bayesian placemap formula over the full 3D space $(x, y, k)$:

$$P(x, y, k \mid \mathbf{n}_t) \propto P(x, y, k) \prod_{i=1}^{N} \text{Poisson}\!\left(n_i^{(t)} \mid \Delta t \cdot f_k^{(i)}(x, y)\right)$$

Because `smooth[-1] = 0.0` and `grid_bin[-1] = 1.0` (no smoothing or interpolation across $k$), the context axis is treated as a **categorical index** — the decoder evaluates each context slice independently using the same spike counts, yielding a joint probability volume of shape $(n_x, n_y, K, T)$.

#### Context marginalisation

The 2D spatial posterior for each context is obtained by marginalizing over the other context:

$$P(k \mid \mathbf{n}_t) = \sum_{x, y} P(x, y, k \mid \mathbf{n}_t)$$

In code this is `marginal_z = np.nansum(p_x_given_n, axis=(0, 1))`, then normalized per time bin. This gives a $K$-dimensional probability vector at every $t$ — the context likelihood — which identifies which epoch-context the neural activity best matches.

The 2D spatial posterior conditioned on the most likely context $k^*$ is then:

$$P(x, y \mid \mathbf{n}_t, k^*) \propto P(x, y, k^* \mid \mathbf{n}_t)$$

which is simply the $k^*$-th slice of the decoded volume.

#### Key difference from Pseudo2D (1D input)

| Property | Pseudo2D (kdiba 1D) | Pseudo3D (NWB/Bapun 2D) |
|---|---|---|
| Input PF dimensionality | 1D ($x$) | 2D ($x, y$) |
| Merged PfND `ndim` | 2 (x, pseudo-y=context) | 3 (x, y, pseudo-z=context) |
| Posterior shape | $(n_x, K, T)$ | $(n_x, n_y, K, T)$ |
| Context axis | pseudo-y | pseudo-z |
| Context marginal | `sum(axis=0)` | `sum(axis=(0,1))` |
| Spatial decode result | 1D position track | 2D spatial map |

## Sources

- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis Pseudo3D OR pseudo3D OR "2D" decoder context merged](/search?q=patterntype:keyword%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20Pseudo3D%20OR%20pseudo3D%20OR%20%222D%22%20decoder%20context%20merged)
- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis NWB session format 2D placefield decoder merged context](/search?q=patterntype:nls%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20NWB%20session%20format%202D%20placefield%20decoder%20merged%20context)
- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis "ndim == 2" OR "pf2D" OR "pseudo3D" merged context decoder NWB](/search?q=patterntype:keyword%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22ndim%20==%202%22%20OR%20%22pf2D%22%20OR%20%22pseudo3D%22%20merged%20context%20decoder%20NWB)
- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis EpochComputationFunctions _build_merged_joint_placefields 2D NWB](/search?q=patterntype:nls%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20EpochComputationFunctions%20_build_merged_joint_placefields%202D%20NWB)
- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis "contextual_pf2D" OR "decode_using_contextual_pf2D" OR "pf2D_decoder" merged context](/search?q=patterntype:keyword%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22contextual_pf2D%22%20OR%20%22decode_using_contextual_pf2D%22%20OR%20%22pf2D_decoder%22%20merged%20context)
- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis "build_contextual_pf2D_decoder"](/search?q=patterntype:keyword%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22build_contextual_pf2D_decoder%22)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis](/github.com/CommanderPho/pyPhoPlaceCellAnalysis)
- [context:@halechr/PhoPhDProjects_CodeOnly def build_contextual_pf2D_decoder|def decode_using_contextual_pf2D_decoder repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ patterntype:regex](/search?q=context:@halechr%2FPhoPhDProjects_CodeOnly%20def%20build_contextual_pf2D_decoder%7Cdef%20decode_using_contextual_pf2D_decoder%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20patterntype:regex)
- [context:@halechr/PhoPhDProjects_CodeOnly build_contextual_pf2D_decoder file:PendingNotebookCode.py repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ patterntype:keyword](/search?q=context:@halechr%2FPhoPhDProjects_CodeOnly%20build_contextual_pf2D_decoder%20file:PendingNotebookCode.py%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20patterntype:keyword)
- [context:@halechr/PhoPhDProjects_CodeOnly repo:CommanderPho/pyPhoPlaceCellAnalysis "build_contextual_pf2D_decoder" OR "decode_using_contextual_pf2D_decoder" count:50](/search?q=patterntype:keyword%20context:@halechr%2FPhoPhDProjects_CodeOnly%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20%22build_contextual_pf2D_decoder%22%20OR%20%22decode_using_contextual_pf2D_decoder%22%20count:50)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@0676630?L5800-5950](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5800-5950)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@0676630?L8280-8380](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8280-8380)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py@0676630?L5950-6060](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/SpecificResults/PendingNotebookCode.py?L5950-6060)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py@0676630?L8380-8460](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@067663033d6f6e309aa7920e6c44e4c246633d0c/-/blob/src/pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py?L8380-8460)
- [github.com/CommanderPho/Spike3DWorkEnv/NeuroPy/neuropy/analyses/placefields.py@eadd004?L1800-1870](/github.com/CommanderPho/Spike3DWorkEnv@eadd004bacb5268fb8e63c6d015268f42c59b726/-/blob/NeuroPy/neuropy/analyses/placefields.py?L1800-1870)

