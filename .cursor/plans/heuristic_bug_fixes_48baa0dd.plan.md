---
name: Heuristic Bug Fixes
overview: Fix concrete bugs in the existing heuristic replay partitioning/scoring pipeline so sequential vs non-sequential separation matches the intended algorithm—without rewriting the approach.
todos:
  - id: fix-dir-jitter
    content: Fix prev_accum_dir=None jitter hack in partition_subsequences_ignoring_repeated_similar_positions
    status: completed
  - id: fix-equiv-groups
    content: Anchor initial_v in find_value_equiv_groups; restore len_excluding_both-first ranking + usable tie-breakers
    status: completed
  - id: fix-intrusion-merge
    content: "Fix intrusion bridging: size rule vs intrusion, direction congruence, iterate intrusions"
    status: completed
  - id: fix-jump-split
    content: Fix off-by-one jump split; stop enforce_max_jump_distance from un-merging bridges
    status: completed
  - id: fix-score-non-intrusion
    content: Compute mseq_tcov/mseq_dtrav on non-intrusion positions only
    status: completed
  - id: fix-tcov-normalize
    content: Normalize track coverage by decoder track length (Long/Short)
    status: completed
  - id: fix-shuffle-thresh
    content: Align shuffle same_thresh_cm with real pipeline scoring
    status: completed
  - id: fix-secondary
    content: Fix jump velocity, avg_jump prepend, HeuristicScoresTuple commas, forwardicity abs_greater
    status: completed
  - id: add-regression-tests
    content: Add focused unit tests for each fixed failure mode
    status: completed
isProject: false
---

# Fix Concrete Heuristic Replay Scoring Bugs

## Scope

Minimal surgical fixes in [`heuristic_replay_scoring.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Analysis\Decoder\heuristic_replay_scoring.py) and related callers. **No DP rewrite**, no new metrics (`N_eff`), no posterior-weighted MAP.

Locked decisions:
- Normalize `mseq_tcov` / coverage by **decoder track length** (Long 214 / Short 144).
- Rank main subsequence by **`len_excluding_both` first**.

## Bugs to fix (in order)

### 1. Direction partitioning: jitter must not erase direction memory

In `partition_subsequences_ignoring_repeated_similar_positions` (~line 679), remove `prev_accum_dir = None` on sub-threshold sign flips. Keep accumulating in the current subsequence; only split when `|Δ| > same_thresh` **and** sign actually reverses relative to the last *accepted* direction.

Desired: small reverse jitter stays inside one subsequence; a real turnaround (`|Δ| > same_thresh`) still splits.

### 2. `find_value_equiv_groups`: anchor `initial_v`

In the `else` branch (~line 581), **do not** set `initial_v = v`. Keep `initial_v` as the start of the plateau until `|v - initial_v| > same_thresh_cm`.

Desired: smooth `+8 cm/bin` sweeps count as many non-repeat bins; stationary plateaus collapse correctly.

### 3. Restore main-subsequence ranking to spatial progress

Change `main_subsequence_ranking_columns` default (~line 202) to:

`['len_excluding_both', 'len_excluding_intrusions', 'len_excluding_repeats', 'len']`

Also drop the dead duplicate `'subsequence_idx'` early in `rank_keys` so `total_distance_traveled` / `track_coverage_score` can break ties (in both `post_compute_subsequence_properties` and `compute_shuffle_scalar_scores`).

### 4. Intrusion bridging: match manuscript intent

In `merge_over_ignored_intrusions` / `_compute_sequences_spanning_ignored_intrusions`:
- Require flanking length vs **intrusion** length (not vs the long non-intrusion subsequence): e.g. at least one flanking side longer than the intrusion (manuscript: flanking exceeds intrusion duration).
- Require **same travel direction** on the two sides being joined before merge.
- Prefer iterating potential intrusions (`len <= max_ignore_bins`) rather than non-intrusions, so symmetric `5+1+5` cases can bridge.

Keep existing jump-distance gate and replace-dict dedupe.

### 5. Jump split: off-by-one + stop un-merging bridges

- In `enforce_max_jump_distance`, split after the jump: use `diff_idx + 1` when converting diff edges to flat split indices.
- Prefer relying on `new_simple_split_main_subsequence_on_max_jump_distance` (already non-intrusion-aware). In `_partition_merge_and_enforce`, stop calling the broken `enforce_max_jump_distance` path that re-splits *including* bridged intrusion bins; keep only the main-sequence non-intrusion jump split after DF rebuild.

### 6. Score bridged subsequences without intrusion bins

In `post_compute_subsequence_properties` and `compute_shuffle_scalar_scores`, compute `total_distance_traveled` and `track_coverage_score` on **`a_subsequence_non_intrusion_pos`**, not the full merged subsequence that still contains bridged outliers.

### 7. Short/Long-fair track coverage

Thread `a_decoder_track_length` into `track_coverage_score` (and callers that build `mseq_tcov`):
- Denominator = decoder track length (214 Long / 144 Short), not shared Long `pos_bin_edges` span.
- Update `bin_wise_track_coverage_score_fn` similarly if it still divides by all shared `n_pos_bins` without Short support adjustment.

Wire length from existing `decoder_track_length_dict` in `compute_all_heuristic_scores` / shuffle scalar path.

### 8. Align shuffle vs real `same_thresh_cm`

Ensure `DecodedEpochScoreShuffle` heuristic shuffle scoring uses the same `same_thresh_cm` as pipeline real scoring (10.7 / 5% of Long), not the default `6.0` in `compute_shuffle_heuristic_score_arrays`.

Likely touch: [`SequenceBasedComputations.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\ComputationFunctions\MultiContextComputationFunctions\SequenceBasedComputations.py) call sites (~786, ~1238) and/or function defaults.

### 9. Secondary clear arithmetic bugs

- `bin_by_bin_position_jump_velocity`: divide by **bin duration** (`Δt`), not absolute `time_window_centers`.
- `avg_jump_cm` / large-jump ratio: exclude the synthetic prepended zero jump; average / count over `N-1` transitions.
- `HeuristicScoresTuple` field defs: remove trailing commas so defaults are scalars, not 1-tuples.
- `SCORE_EXTREME_MODE['forwardicity']`: `'abs_greater'` so reverse replays are not treated as non-extreme.

### 10. Filtering consistency (paper path)

Keep / prefer same-decoder conjunction (`HeuristicThresholdFiltering.df_is_included_criteria_fn`). Where FigureS8 / notebook predicates still use independent `HIGHEST_mseq_len` AND `HIGHEST_mseq_tcov`, point them at the same-decoder predicate (or document that `HIGHEST_*` alone is insufficient)—minimal change only where that predicate is actively used for heuristic-passing overlays.

## Tests

Extend [`test_heuristic_sequence_partitioning.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_heuristic_sequence_partitioning.py) with focused cases (no full rewrite):
- Sub-threshold reverse jitter does not split; above-threshold turnaround does.
- `find_value_equiv_groups` anchors plateau start (smooth sweep → many groups; flat plateau → one).
- Symmetric `5+1+5` same-direction merge; opposite-direction sides do not merge.
- Jump split index places outlier in its own fragment; bridged outlier does not inflate `mseq_tcov`/`mseq_dtrav`.
- Short vs Long full-traversal `mseq_tcov` both near 1.0 when normalized by track length.

## Out of scope

DP monotonic path finder, participation-ratio `N_eff`, posterior-weighted scoring, velocity-scaled `max_jump` redesign beyond fixing the existing jump-split bugs.