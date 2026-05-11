---
name: Bapun RTC notebook
overview: "Add a new tutorial-style notebook under `replay_trajectory_classification` that mirrors [`04-Classifying_with_Sorted_Spikes.ipynb`](h:\\TEMP\\Spike3DEnv_ExploreUpgrade\\Spike3DWorkEnv\\replay_trajectory_classification\\notebooks\\tutorial\\04-Classifying_with_Sorted_Spikes.ipynb) but ingests real Bapun multi-maze data: load the session via NeuroPy’s Bapun loader, build high-rate `position` + binary `spikes` tensors expected by `SortedSpikesClassifier`, then run the same classifier / predict flow as notebook 04. Default example targets Day5TwoNovel-style data with 1D linearized coordinates from `*.maze1.linear.npy` (maze1 epoch only for clarity)."
todos:
  - id: add-notebook
    content: "Create `notebooks/tutorial/06-Classifying_Bapun_Sorted_Spikes.ipynb` with markdown + cells: NeuroPy Bapun load, linear.npy load, paradigm mask, resample, spike binning, mirror 04 classifier + predict/plot."
    status: completed
  - id: adapter-linear-npy
    content: Implement robust unpack of `maze1.linear.npy` (inspect pickle structure once; normalize to t + 1D position arrays with clear errors).
    status: completed
  - id: neuropy-prereq-doc
    content: "Top-of-notebook documentation: required Bapun files, GEN linear npy prerequisite, NeuroPy install from peer path / env note."
    status: completed
  - id: smoke-run
    content: Execute notebook against an available Bapun Day5TwoNovel-style folder; fix any API mismatches (paradigm columns, neuron_type filters).
    status: completed
isProject: false
---

# Bapun + SortedSpikesClassifier notebook

## Context

- **Target tutorial**: [`notebooks/tutorial/04-Classifying_with_Sorted_Spikes.ipynb`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\notebooks\tutorial\04-Classifying_with_Sorted_Spikes.ipynb) — uses `SortedSpikesClassifier`, a 3×3 `continuous_transition_types` matrix (`RandomWalk` / `Uniform` / `Identity`), `Environment(place_bin_size=...)`, `fit(position_1d, spikes)`, then `predict` on synthetic replay snippets from `sorted_spikes_simulation`.
- **Required RTC tensor shapes** (from [`01-Introduction_and_Data_Format.ipynb`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\notebooks\tutorial\01-Introduction_and_Data_Format.ipynb) and [`classifier.py` `fit`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\replay_trajectory_classification\classifier.py)): `position` shape `(n_time, n_position_dims)`; `spikes` shape `(n_time, n_neurons)` with **dense small time bins (typically 1–2 ms)**.
- **Bapun “known format”** is implemented in NeuroPy as [`BapunDataSessionFormatRegisteredClass`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\Specific\BapunDataSessionFormat.py): required `{}.xml`, `{}.neurons.npy`, `{}.position.npy`, `{}.paradigm.npy`, `{}.probegroup.npy`. Commented file tree also lists **`{}.maze1.linear.npy` / `{}.maze2.linear.npy` as GEN** ( produced by the usual linearization workflow — see NeuroPy [`examples/Placefield Analysis.ipynb`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\examples\Placefield%20Analysis.ipynb)).
- **Repo dependency gap**: [`pyproject.toml`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\pyproject.toml) does **not** include `neuropy`. The notebook will treat NeuroPy as an **optional peer install** (editable path or workspace package), documented at the top.

## User choice (locked)

- **Default data scenario**: multi-maze day (Day5TwoNovel-style) using **1D linear position** from `*.maze1.linear.npy` during the **`maze1` paradigm epoch** (single-environment setup, parallel to notebook 04). A short markdown note can describe extending to maze2 or `environment_labels` without implementing full multi-env fitting in v1.

## Notebook deliverable

- **New file**: [`notebooks/tutorial/06-Classifying_Bapun_Sorted_Spikes.ipynb`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\notebooks\tutorial\06-Classifying_Bapun_Sorted_Spikes.ipynb) (06 follows existing `01`–`05` sequence).

### Section outline (cells)

1. **Prerequisites / paths**
   - Configure `BASEDIR` (session folder containing the Bapun tree).
   - State explicitly that `maze1.linear.npy` must exist (or point to NeuroPy example / your pipeline that generates it).
   - Note `uv sync` / install `replay_trajectory_classification` plus **manual** NeuroPy install from the sibling repo.

2. **Load Bapun session (canonical)**
 - `from pathlib import Path`
   - `from neuropy.core.session.Formats.Specific.BapunDataSessionFormat import BapunDataSessionFormatRegisteredClass`
   - `session = BapunDataSessionFormatRegisteredClass.build_session(BASEDIR)`
   - `session, _ = BapunDataSessionFormatRegisteredClass.load_session(session)`
   - This populates `session.neurons`, `session.position`, `session.paradigm` using the same loaders as the rest of your stack ([`_load_neurons_file`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\core\session\Formats\Specific\BapunDataSessionFormat.py), etc.).

3. **Load linearized track for maze1**
   - Resolve `session_name` stem (same helper pattern as format class: sole `.xml` in folder).
   - `linear_path = BASEDIR / f"{session_name}.maze1.linear.npy"`; **fail with a clear message** if missing.
   - `np.load(..., allow_pickle=True)` and unpack whatever container NeuroPy saves (dict-like / custom object) into **`t_seconds`** and **`linear_position`** arrays (inspect interactively in the first implementation pass; keep a small “adapter” cell that normalizes to `(t_lin, x_lin)`).

4. **Epoch mask**
   - Read `maze1` interval from `session.paradigm` / `Epoch` dataframe (labels align with Bapun conventions already used in your other notebooks).
   - Restrict linear position samples to that interval; optionally build `is_training` True only for **running / high-speed** bins later (start with full maze1 for simplicity).

5. **Resample to classifier clock**
   - Choose `sampling_frequency` (start with **1000 Hz** to match [`sorted_spikes_simulation.SAMPLING_FREQUENCY`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\replay_trajectory_classification\sorted_spikes_simulation.py)); document lowering to 500 Hz if memory is an issue.
   - Build uniform `time` grid covering the maze1 window.
   - Interpolate **1D linear coordinate** onto `time` via `numpy.interp` (or `scipy.interpolate.interp1d` if you need extrapolation policy).

6. **Spike tensor**
   - From `session.neurons` / `spikes_df`, filter **pyramidal** units (drop `mua`, map mislabeled `'1'` → `'mua'` the same way the loader comments describe).
   - For each retained unit, histogram spike times into `time` bins → binary `(n_time, n_neurons)` **via `np.minimum(count, 1)`** to match RTC’s binary expectation.

7. **Classifier section (mirror 04)**
   - Reuse imports and structure from 04: `SortedSpikesClassifier`, `Environment`, `RandomWalk`, `Uniform`, `Identity`, same `continuous_transition_types` pattern and `state_names = ['continuous', 'fragmented', 'stationary']`.
   - Replace `movement_var` / `place_bin_size` heuristics with values appropriate for **real linearized track extent** (e.g. scale from empirical position variance or a sensible default with a markdown note to tune).
   - `classifier.fit(linear_position_column, spikes, is_training=...)` if you add speed-based masking; else omit.

8. **Prediction / plots**
   - Either reuse 04’s `make_continuous_replay` / `make_hover_replay` demos **purely to visualize classifier behavior**, or add an optional cell: load a short ripple/PBE window if `*.ripple.npy` / `*.pbe.npy` exists (optional, guarded by `Path.is_file()`).

### Implementation notes

- Keep logic **in the notebook** initially (no new `.py` modules required); add a helper function in-cell for “Bapun → RTC arrays” to avoid duplication.
- Follow your Python style rules: **single-line calls/signatures where reasonable**, **two blank lines between class methods** if any small helper class is introduced (unlikely in v1).
- After adding the notebook, optionally add a one-line pointer in [`CLAUDE.md`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\CLAUDE.md) “Example notebooks” — only if you want discoverability (user did not request doc edits; default is **skip** unless you ask).

## Validation

- Smoke-test by running the notebook locally with a real session on disk (your `H:\Data\Bapun\...` or `W:\data\Bapun\...` trees).
- Expect Cupy GPU warning messages; they are normal per [`CLAUDE.md`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\replay_trajectory_classification\CLAUDE.md).
