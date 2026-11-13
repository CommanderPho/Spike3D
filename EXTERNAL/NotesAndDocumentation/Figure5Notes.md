# Figure 5 Notes — Measured vs Decoded Occupancy

Collected notes, realizations, and plan decisions for paper **Figure 5** (measured vs decoded occupancy). Sources: paper notebook markdown/code comments, Cursor plans, and related chat summaries.

---

## Primary notebook notes

**Source:** `EXTERNAL/PhoDibaPaper2024Book/PhoDibaPaper2024_FULL_ARCHIVE_2026-08-31.ipynb`

### Section header

> ## 🖼️ Figure 5 - Meas. v. Decoded Occupancies

### 2026-10-22 — track x-position misalignment (core realization)

Embedded in the Figure 5 H5/FAT compute cell (`run-fig-5`):

> ### 2026-10-22 - NOTE: Actually all sessions sit at minorly different x-positions (tracks positioned slightly differently). Using the fixed bin positions (or ones zeroed around their extent by session) seems better than a fake mapping, right?

Implication: do **not** invent a cross-session cm remapping. Prefer **per-session bin indices** (`split_by_session=True`).

### Preferred compute flags (same cell)

- `use_most_likely_pos_only_pseudo_occupancy = False`
- `use_all_decoders_simultaneously_occupancy = True` — “treats all decoders as always running. Doesn't care about the most likely or anything. Looks good.”
- `split_by_session = True` — keep occupancy per session; `Figure5.main` plots one 2×3 subfigure per session

### 2026-11-12 — CSV-based Figure 5 path

> # 2026-11-12 - New Figure 5 from `*_(fig5_occupancy)_*.csv` files

Batch exports `(fig5_occupancy)` long CSVs into `collected_outputs`; paper notebook loads them via `Figure5.load_occupancy_long_csvs_to_session_nested_dicts` and plots with `split_by_session=True` (by session and by animal aggregation).

### Older / alternate path (superseded for paper assembly)

> #### ❎ 2025-11-05 - meas_v_decoded_occupancy from FAT_df

Across-session FAT-based soft/hard weighted histograms of `most_likely_positions_1D` — kept as an alternate attempt, not the preferred paper path.

---

## Realizations summary (from notebook + follow-on work)

### 1. Cross-session aggregation smears position

Sessions sit at slightly different track x-positions. Summing occupancy across sessions blends those offsets and smears bins. Prefer **per-session** occupancy / plots (`split_by_session=True`) instead of a pooled aggregate.

### 2. Prefer bin indices over a fake cm remapping

Rather than inventing a cross-session position remapping, keep **fixed bin indices** (or zero occupancy around each session’s own extent). That matches how decoded occupancy is already stored and avoids a fake shared-cm axis.

### 3. Simultaneous decoder occupancy looks better

`use_all_decoders_simultaneously_occupancy=True` (treat all four decoders as always running; don’t gate on most-likely) looks better vs most-likely-only / non-simultaneous modes. Related: the four posteriors are **independent 1D decoders**, not a true Pseudo2D joint.

### 4. Measured occupancy is decode-bin `binned_x_meas`, not PF occupancy

Figure 5’s measured curve should come from decode-bin-aligned `binned_x_meas` (interpolated onto decode bin centers), histogrammed into the same position-bin vectors — **not** classic placefield `pf.occupancy`. `x_meas` / `binned_x_meas` is only non-NA for `masked_time_bin_fill_type='nan_filled'`.

**Bottom line:** Figure 5 should compare measured vs decoded occupancy **per session, on shared bin indices**, with **simultaneous** decoded occupancy as the preferred decoded definition.

---

## Design decision: batch pipeline export (preferred over H5+FAT rebuild)

With live session pipelines available during batch, do **not** rebuild Figure 5 from full H5 posteriors + FAT for the paper figure.

### Why not the H5 path (legacy)

The H5/FAT notebook path existed because batch only left posteriors + FAT. That forced:

- recovering measured `x` indirectly (`binned_x_meas`, only on `nan_filled`)
- fighting track misalignment across sessions
- re-deriving occupancy from stacked independent 1D posteriors

### Recommended design (implemented)

1. **Per-session batch completion** via `export_decoded_variables_evaluation_completion_function` → `Figure5.export_session_occupancy_csv`
2. **Tiny artifact:** long-form `*(fig5_occupancy)_tbin-*.csv` under `collected_outputs` (not full posteriors)
3. **Paper notebook only stacks:** load CSVs → `Figure5.main(..., split_by_session=True)`

Concrete choices:

| Choice | Decision |
|--------|----------|
| Geometry | Occupancy on **bin index per session** — never shared cm axis |
| Measured | Decode-bin-aligned `binned_x_meas`, not `pf.occupancy` |
| PBEs | Plot decoded; measured may be empty / N/A (animal not running track) |
| Decoded | Simultaneous occupancy (`nansum` of \(P(x\mid n,d)\) over time × `time_bin_size`) |
| Batch figures | Optional; CSV is the contract |

### Decoded occupancy accrual (CSV path)

Soft / probability-weighted, not MAP bin counts:

1. Flatten Pseudo2D → `timebins_p_x_given_n` shape `(n_pos, n_decoders, n_t)`
2. For each decoder \(d\) and period mask: `occ_seconds = nansum(P[:, d, t_mask], axis=-1) * time_bin_size`
3. Normalize for `occupancy_frac`
4. Decoders stay separate (`source` = decoder name); no winner-takes-all decoder step

Measured rows: hard counts of `binned_x_meas`, then × `time_bin_size`.

---

## Cursor plans (Fig5-related)

### `fig5_split_by_session_3a899231.plan.md` — completed

Add `split_by_session` so each session keeps its own occupancy vectors and is drawn as a stacked 2×3 matplotlib subfigure, instead of summing misaligned tracks.

- Default `False` preserves aggregated cells; Figure 5 notebook cell passes `True`
- Session keys: `a_sess_ctxt.get_description(separator='_')` to match FAT `session_name`
- Out of scope: new x-position remapping / zeroing-by-extent (per-session bin index **is** the fix)

### `fig5_occupancy_batch_export_2dc11178.plan.md` — completed

Extend the existing uncertainty completion hook so each session also writes `(fig5_occupancy)` CSV into `collected_outputs` from live `EpochComputations` + measured position — without touching posterior H5.

Chosen defaults:

- Long-form `.csv` (not H5), parseable by `find_most_recent_files`
- Decoded: simultaneous, decoder-split from EpochComputations Pseudo2D flatten
- Measured: `_helper_add_interpolated_position_columns_to_decoded_result_df` on that session’s `xbin`
- Epochs: `laps` and `pbe`, `masked_time_bin_fill_type='nan_filled'`, `trained_compute_epochs='laps'`
- Periods: `pre-delta`, `post-delta`, `all`
- Separate try/except so Fig5 failure does not erase uncertainty (and vice versa)
- Out of scope: cross-session cm remapping / aggregated (non-split) Fig5; changing posterior H5 export

### `fix_gl_csv_exports_db5e7c74.plan.md` — completed

Hardened GL export failures for both uncertainty and Fig5 CSVs:

| Export | Symptom | Cause |
|--------|---------|--------|
| uncertainty | `too many values to unpack (expected 2)` | Stale 2-value unpack of helper (returns one DF) |
| fig5_occupancy | length mismatch `N != M` | Helper measured cols from mismatched `a_result` vs marginal DF grain |

Fixes: align single-context lookup on posterior key; Fig5 re-fetch on length mismatch before flatten assert; `delta_aligned_start_t` fallback; keep helper `t_bin_center` fallback.

---

## Code / validation history touchpoints

| Item | Location / note |
|------|-----------------|
| `Figure5` class | `PhoDiba2023Paper.py` — extracted 2026-10-22; tags `figure-5`, `fig-5` |
| `MeasuredVsDecodedOccupancy` | `PendingNotebookCode.py` — 2025-05-15 validation Kamran requested (“showed the expected result”) |
| Batch hook | `export_decoded_variables_evaluation_completion_function` — uncertainty + Fig5 CSV; needs `EpochComputations` |
| `--only` recipe | `ProcessBatchOutputs_qclus1246789_Only.ipy` — exports context-uncertainty + Fig5 occupancy CSVs |
| Review notebook | `ReviewOfWork_2026-04-01.ipynb` — `## 2025-05-15 - Fig. 5 - Decoded vs Measured Occupancy` + publication folder `"Figure 5 - Measured v Decoded Occupancy"` |
| File type loader | `AcrossSessionResults` — `file_type == 'fig5_occupancy'` concatenates long CSVs |

### Export filename pattern

```text
{date}-{session}__{custom_replay}-(fig5_occupancy)_tbin-{time_bin_size}.csv
```

Written under `collected_outputs` via `build_complete_session_identifier_filename_string`.

---

## What not to do (settled)

- Export full `p_x_given_n` solely for Figure 5
- Aggregate sessions onto a shared cm axis
- Treat classic placefield occupancy as the measured curve for this comparison
- Remove/change `(decoded_posteriors)` H5 export (still needed elsewhere)
- Let Fig5 CSV failure erase uncertainty export results (or vice versa)
