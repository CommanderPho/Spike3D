---
name: Init sampling unit
overview: Move `sampling_unit` onto `PrePostDeltaDistributionComparison.__init__` so cluster modes skip lap grouping and the mixed pre/post-within-epoch check that currently fails on PBE time bins.
todos:
  - id: init-kwarg
    content: Add sampling_unit to __init__; cluster path skips lap grouping/checks and caches session pre/post arrays
    status: completed
  - id: bootstrap-effects
    content: Point cluster bootstrap and _session_effect_rows at session caches; run() uses self.sampling_unit
    status: completed
  - id: docs-compat
    content: Update docstrings; run(sampling_unit=) must match init or raise
    status: completed
isProject: false
---

# Know `sampling_unit` in `__init__`

Add `sampling_unit='block'` to [`PrePostDeltaDistributionComparison.__init__`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) and branch setup there. Cluster modes (`'animal'` / `'session'`) never build `_lap_id`, `_lap_table`, `_runs`, or `_values_by_lap`.

Notebook usage becomes:

```python
PrePostDeltaDistributionComparison(df, sampling_unit='session', n_bootstrap=10000, random_state=1337)
comparison.run(perform_block_permutation_test=True)
```

## Init split

Shared for all units: filter to pre/post labels; require `value_col`, `condition_col`, `animal_col`, `session_col`; numeric finite `P_Short` in `[0, 1]`; factorize animals; require every animal has both conditions.

**`'block'`** (unchanged): also require `lap_col` / `lap_order_col`, mixed-condition and constant-order checks, lap caches, chronological runs, `_sessions_by_profile`.

**`'animal'` / `'session'`:** do not require lap columns. Cache pre/post value arrays per `(animal_code, session)` and build `_sessions_by_profile` from which conditions appear in each session. Skip the “each lap has one condition” error (PBE bins that straddle delta are just session members).

`block_length_laps` stays required only for `'block'`.

## Downstream

- `_hierarchical_bootstrap`: cluster path copies those session pre/post arrays; no lap-id walk. Block path unchanged.
- `_session_effect_rows`: group the session caches, not `_runs`.
- `run(sampling_unit=None)`: default to `self.sampling_unit`. If a value is passed and it differs from init, raise (init already chose the data layout). Drop the redundant validation that currently lives only in `run()`.

Update the class / `run()` docstrings: cluster modes ignore parent-epoch identity; pass `sampling_unit` to the constructor.

Do not edit notebooks. Existing `run(sampling_unit='session')` still works only if the constructor was also built with `'session'` (or `'animal'`).
