---
name: PLI auto 1D dispatch
overview: Keep the explicit `*_1d` implementations, but make the main 2D entry points (`calculate_pli_score`, `compute_and_plot_posterior_stack`, `filter_to_position_like_epochs_only`) detect 1D inputs and forward to the 1D variants so callers need not pick the API manually.
todos:
  - id: detect-helper
    content: Add _should_use_1d_pli detection helper with (n_x,n_y) vs (n_pos,n_t) rules
    status: completed
  - id: dispatch-score-stack
    content: Auto-dispatch calculate_pli_score and compute_and_plot_posterior_stack; alias is_position_like on 1D DF
    status: completed
  - id: dispatch-filter
    content: Auto-dispatch filter_to_position_like_epochs_only when ybin None / 1D posteriors; remap default 0.42 cutoff to 1D default
    status: completed
  - id: docs-tests
    content: Update class/entry docstrings and extend test_position_like_posterior_scoring_1d.py for dispatch
    status: completed
isProject: false
---

# Auto-dispatch 1D PLI from 2D entry points

## Goal

Callers can keep using the existing 2D names; when the posterior is 1D (BATCH `(n_pos, n_t)` / `(n_pos,)`), those functions forward to the 1D implementations already on [`PositionLikePosteriorScoring`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py). Explicit `*_1d` methods stay as the real 1D implementations.

## Detection helper (new)

Add `@classmethod _should_use_1d_pli(cls, arr, x_edges, y_edges=None) -> bool` next to the 1D section:

- `y_edges is None` → **1D**
- `len(y_edges) <= 2` (degenerate single y-bin) → **1D**
- `arr.ndim == 1` → **1D**
- `arr.ndim == 2`:
  - shape `(len(x_edges)-1, len(y_edges)-1)` → **2D** single spatial frame
  - shape `[0] == len(x_edges)-1` and `[1] != len(y_edges)-1` → **1D** stack `(n_pos, n_t)` (BATCH layout)
  - shape `(n_pos, 1)` with matching x → **1D**
- `arr.ndim == 3`: **1D** only if `shape[1] == 1` (degenerate y); else **2D** stack
- otherwise → **2D** (raise later if shape is invalid for 2D math)

## Entry-point changes

### 1. `calculate_pli_score` (~7674)

- Make `y_edges: Optional[...] = None`.
- At the top, if `_should_use_1d_pli(posterior, x_edges, y_edges)`:
  - map kwargs: pass through `w_sharpness`, `w_locality`, `noise_threshold_percent`; map `sigma_tolerated_env_diagonal_percent` → only used on 2D path (do **not** silently reinterpret as track-length %); 1D uses its own `sigma_tolerated_track_length_percent_1d` default unless caller passed kwargs meant for 1D via `**` later.
  - `return cls.calculate_pli_1d_score(posterior, x_edges=x_edges, w_sharpness=..., w_locality=..., noise_threshold_percent=...)`
- Else keep existing 2D body (unchanged math).
- Docstring: note auto-dispatch; 1D cutoffs/geometry still come from the 1D path.

### 2. `compute_and_plot_posterior_stack` (~8023)

- Make `y_edges: Optional[...] = None`.
- **Before** the current `ndim == 2 → [:, :, None]` expansion, if `_should_use_1d_pli(...)`:
  - `return cls.compute_and_plot_posterior_stack_1d(posterior_stack, x_edges=x_edges, noise_threshold_percent=..., should_plot_results=..., max_n_frames_to_plot=..., cols=..., debug_print=..., **pli_score_kwargs)`
- After that return, ensure callers of the 2D stack API still see `is_position_like`: in `compute_and_plot_posterior_stack_1d` (or in the dispatch wrapper), set `is_position_like = is_position_like_1d` in addition to `is_position_like_1d` so existing column checks keep working.
- Else existing 2D path unchanged.

### 3. `filter_to_position_like_epochs_only` (~8108 area)

- Make `ybin: Optional[...] = None`.
- After loading/`concatenate` prep, inspect the first epoch (or flattened array): if 1D:
  - Score via `compute_and_plot_posterior_stack_1d` (or per-epoch) with `x_edges=xbin`, `position_like_score_cutoff_1d=position_like_score_cutoff` when the caller passed a cutoff (use 1D class default if they pass the old 2D default `0.42`? **Concrete choice:** if `ybin is None` or detected 1D, interpret `position_like_score_cutoff` against **1D** cutoffs: when the argument is still the 2D default `0.42`, substitute `cls.position_like_score_cutoff_1d` so accidental reuse of the 2D default does not under-threshold; if the caller passes any other value, use it as-is on the 1D path.
  - Build `is_position_like_list` from `is_position_like_1d` / `is_position_like` and keep the existing mask/drop logic.
- Else keep current 2D `axis=2` concatenate + `compute_and_plot_posterior_stack` path.

### 4. Docs

- Update class docstring: primary APIs auto-dispatch; `*_1d` remain explicit/advanced.
- Soften “use `*_1d` instead” notes on `calculate_pli_score` / filter to “auto-dispatches; `*_1d` still available”.

## Tests

Extend [`tests/test_position_like_posterior_scoring_1d.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_position_like_posterior_scoring_1d.py):

- `calculate_pli_score(P_1d, x_edges)` with `y_edges=None` matches `calculate_pli_1d_score`
- `calculate_pli_score` on true `(n_x, n_y)` with both edges stays on 2D path (delta peak score near 1 under 2D math)
- `compute_and_plot_posterior_stack((n_pos, n_t), x_edges)` without `y_edges` returns `n_t` rows and sets both `is_position_like` and `is_position_like_1d`
- `_should_use_1d_pli` cases: `(n_x,n_y)` false; `(n_pos,n_t)` true when y missing or y length mismatch; `(n_x,1,n_t)` true

## Files

- [`PendingNotebookCode.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\SpecificResults\PendingNotebookCode.py) — helper + dispatch in the three entry points; small DF column alias in 1D stack
- [`tests/test_position_like_posterior_scoring_1d.py`](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\tests\test_position_like_posterior_scoring_1d.py) — dispatch tests
