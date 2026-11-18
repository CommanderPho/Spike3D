---
name: keyed posterior overlays
overview: Extend the VisPy 3D time-series viewer to support multiple keyed posterior image planes at arbitrary time values, with API methods for add/get/toggle/remove while preserving existing slider behavior.
todos:
  - id: add-overlay-state
    content: Add keyed overlay state fields on Volumentric2DTimeSeriesPlotter dataclass.
    status: completed
  - id: extract-plane-helpers
    content: Extract shared helper logic for 2D posterior normalization, RGBA conversion, and image-plane transform placement from time_value.
    status: completed
  - id: implement-overlay-api
    content: Implement add/get/list/toggle/remove/clear keyed decoded posterior methods with deterministic auto key generation.
    status: completed
  - id: keep-slider-compat
    content: Keep existing active slider posterior plane behavior unchanged and independent from keyed overlays.
    status: completed
  - id: refresh-scene-tree
    content: Ensure scene tree refreshes after overlay add/remove operations.
    status: completed
  - id: lint-and-smoke-check
    content: Run lints for modified file and verify multiple overlay workflows work as expected.
    status: completed
isProject: false
---

# Add Keyed Posterior Overlays

## Goal

Enable arbitrary decoded posterior planes to coexist in the 3D scene, each indexed by a unique string key (auto-generated when omitted), positioned by explicit `time_value`, and controllable through retrieval/visibility APIs.

## Files To Change

- [h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/predicitive_decoding_vispy.py)

## Planned Changes

- Add state to `Volumentric2DTimeSeriesPlotter` for keyed overlays:
  - `decoded_posteriors_by_key: Dict[str, Dict[str, Any]]` (metadata + visual handle)
  - `decoded_posterior_counter: int` (deterministic auto key generation)
- Refactor posterior rendering internals:
  - Keep current active-slider posterior path intact for backward compatibility.
  - Extract reusable helper(s) for:
    - validating/normalizing 2D posterior arrays
    - colormap-to-RGBA conversion
    - constructing/translating `vz.Image` plane at z from `time_value` via `z_val = (time_value - self.t_min) * self.z_scale`
- Add public keyed overlay API methods (minimal, explicit behavior):
  - `add_decoded_posterior(decoded_posterior_2d, time_value, unique_identifier=None, visible=True, replace_if_exists=True) -> str`
  - `get_decoded_posterior(unique_identifier) -> Optional[Dict[str, Any]]`
  - `set_decoded_posterior_visibility(unique_identifier, is_visible: bool) -> bool`
  - `remove_decoded_posterior(unique_identifier) -> bool`
  - `clear_decoded_posteriors()`
  - `list_decoded_posterior_keys() -> List[str]`
- Ensure scene-tree integration is coherent:
  - Give each overlay visual a distinct `name` containing its key.
  - Rebuild/refresh scene tree after add/remove operations to reflect current nodes.
- Preserve existing UX:
  - `update_active_t_bin(...)` continues to update the primary slider-driven `self.posterior_plane` exactly as before.
  - Keyed overlays are independent layers (not removed by slider updates).

## Validation

- Run lint diagnostics on modified file and resolve introduced issues.
- Smoke-check expected behavior manually:
  - add multiple keyed overlays at different `time_value`s
  - toggle one overlay off/on without affecting others
  - retrieve metadata/visual handle by key
  - remove one overlay while keeping remaining overlays + slider-driven plane functional.

