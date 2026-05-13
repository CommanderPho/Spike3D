---
name: Empirical Sprint Detection
overview: Add a new `EmpericalSprintDetection` class at the end of [MovementBurstDetection.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\MovementBurstDetection.py) that estimates running vs stopped from position using data-driven thresholds (GMM on log-speed, Otsu, robust fallback), optional rolling-window “local” thresholds, hysteresis, and duration debouncing—without changing existing detector behavior.
todos:
  - id: add-class-bottom
    content: Add EmpericalSprintDetection + helpers before `if __name__` in MovementBurstDetection.py
    status: completed
  - id: wire-methods
    content: Implement gmm / otsu / robust thresholding, optional rolling_window_s, hysteresis + dwell times
    status: completed
  - id: detect-return
    content: Return dict with intervals, masks, thresholds, and reuse preprocess + compute_movement_features via OptimizedMovementBurstDetector
    status: completed
isProject: false
---

# EmpericalSprintDetection class

## Placement

Insert the new class **immediately before** [`if __name__ == "__main__":`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\MovementBurstDetection.py) (after `example_usage()`), so it lives at the **bottom of the module** as requested and stays grouped away from `BurstAnalyzer` / `OptimizedMovementBurstDetector`. The `if __name__` block stays last.

**Naming:** Use the exact class name `EmpericalSprintDetection` as in your request (common typo for “Empirical”; can add a one-line alias `EmpiricalSprintDetection = EmpericalSprintDetection` if you want both spellings in imports).

## Behavior (methods from prior discussion)

| Mode | Role |
|------|------|
| **`gmm` (default)** | Fit `sklearn.mixture.GaussianMixture(n_components=2)` on `log(speed_smooth + δ)` with small `δ` derived from data. Label the component with **lower mean** as “stopped”. Derive a **single speed cutoff** on the original speed axis by scanning a fine grid of speeds and taking the smallest speed where P(moving) ≥ 0.5 (or midpoint between mixture means mapped back—grid is clearer). |
| **`otsu`** | One-dimensional Otsu on `log(speed_smooth + δ)` implemented in pure NumPy (no new deps); map threshold back to speed as `exp(t_log) - δ`. Use when you want a histogram valley without mixture fitting. |
| **`robust`** | Noise-floor style: e.g. `θ = k * median(speed_smooth)` over the **lower** portion of the distribution (e.g. samples with speed below the session median) **or** `k × MAD(speed_smooth)` on that subset—document the exact formula in the docstring. Used as **fallback** if GMM fails or components are poorly separated (e.g. identical means or near-zero weight on one component). |

**“Local” threshold:** Optional constructor arg `rolling_window_s: Optional[float] = None`. If set (e.g. 30–120 s), for each time sample (or overlapping windows stepped by ~0.5–1 window) refit the chosen method on `speed_smooth` inside the window and assign a **time-varying** threshold (interpolate linearly in time between window centers). If `None`, use one session-global threshold.

**Hysteresis / stability:** Optional `hysteresis_ratio` (e.g. 0.1–0.2): define `θ_run = θ * (1 + r)`, `θ_stop = θ * (1 - r)` so enter/exit running uses different cutoffs. Apply **minimum dwell** with `min_stop_duration_s` and `min_run_duration_s` (constructor defaults similar in spirit to existing burst/rest knobs but smaller, e.g. 0.15–0.3 s) to turn frame-wise labels into intervals.

## Signal pipeline (reuse existing code)

Avoid duplicating smoothing logic: internally instantiate a lightweight [`OptimizedMovementBurstDetector`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\MovementBurstDetection.py) (or call its methods via a throwaway instance) with `velocity_smoothing` passed through, then:

1. `df_clean = detector.preprocess_trajectory(pos_df)`
2. `features_dict = detector.compute_movement_features(df_clean)`
3. Use `features_dict['speed_smooth']`, `features_dict['t']`, `features_dict['dt']`

This keeps units and smoothing consistent with the rest of the file.

## Public API

- **`__init__(self, velocity_smoothing=0.1, threshold_method: Literal['gmm','otsu','robust']='gmm', rolling_window_s=None, hysteresis_ratio=0.15, min_run_duration_s=0.3, min_stop_duration_s=0.15, robust_k=3.0, gmm_random_state=0, ...)`** — keep signature on one line where possible per project rules; use two blank lines between methods.

- **`detect(self, pos_df) -> dict`** returning at least:
  - `'threshold_speed'` — scalar if global, or `'threshold_speed_t'` array aligned with `t` if local mode
  - `'theta_stop'`, `'theta_run'` (after hysteresis) for transparency
  - `'is_running'` — boolean array (post debounce) aligned with `features_dict['t']`
  - `'run_intervals'` / `'stop_intervals'` — list of `{start, end, duration}` (and optionally `mean_speed`)
  - `'sprints'` or `'bursts'` — alias of run intervals for parity with mental model (document that these are **empirical run bouts**, not BOCD bursts)
  - `'features'`, `'processed_data'` — optional echo for plotting/debug

- **Private helpers** (module-level functions or `_` methods): `_log_speed(s, delta)`, `_threshold_gmm`, `_threshold_otsu`, `_threshold_robust`, `_separation_ok`, `_apply_hysteresis_and_debounce`.

## Imports

Add at top of file (with existing sklearn import): `from sklearn.mixture import GaussianMixture` (or import inside the method to avoid import-order noise—prefer top-level next to `DBSCAN`).

## Testing / docs

- No requirement to change `if __name__` block in the first pass; optionally add a 5-line commented example under the class docstring showing `EmpericalSprintDetection(...).detect(pos_df)`.
- No new package dependencies (sklearn and numpy already present per [pyproject.toml](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\pyproject.toml)).

## Out of scope

- Do not refactor `OptimizedMovementBurstDetector` or `compute_movement_trajectories_from_bursts` unless you later ask to wire this class into laps pipelines.
- No notebook edits.
