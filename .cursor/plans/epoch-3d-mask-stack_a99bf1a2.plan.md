---
name: epoch-3d-mask-stack
overview: Add optional Silx 3D stack rendering of epoch masks aligned with time bin 0 in the existing Epoch3DSceneTimeBinViewer.
todos:
  - id: add-config-flags
    content: Add viewer params and state fields to control and hold the 3D mask stack item in Epoch3DSceneTimeBinViewer.
    status: completed
  - id: build-mask-volume
    content: Implement helper to construct a 3D mask volume per epoch from self.plots_data.peak_contours['epoch_prom_t_bin_high_prob_pos_mask'] with documented axis ordering.
    status: completed
    dependencies:
      - add-config-flags
  - id: create-stack-item
    content: Create a ScalarField3D-style item from the epoch mask volume, align it in XY with t_bin_idx=0, and translate it by negative Y offset, adding it to the SceneWindow.
    status: completed
    dependencies:
      - build-mask-volume
  - id: wire-lifecycle
    content: Hook stack creation and cleanup into on_epoch_changed and _clear_time_bin_items so the stack updates correctly when epochs change.
    status: completed
    dependencies:
      - create-stack-item
  - id: api-docs
    content: Expose a simple toggle API or param usage pattern for enabling the 3D mask stack and document coordinate/offset behavior in comments.
    status: completed
    dependencies:
      - wire-lifecycle
---

# Add Optional 3D Mask Stack per Epoch in Epoch3DSceneTimeBinViewer

### Goal

Extend `Epoch3DSceneTimeBinViewer` (in [`pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py`](pyphoplacecellanalysis/GUI/Silx/EpochTimeBinViewerWidget.py)) to optionally render a **3D stack of all masks** for the current epoch:

- The stack should be aligned in XY with the existing `t_bin_idx = 0` time bin surfaces.
- The whole stack should be translated **downward along Y** by a fixed offset (we'll base this on `y_extent` to emulate `y = -SIZE`).

### High-level design

- Reuse the **per-epoch mask data** already computed and stored in `self.plots_data.peak_contours['epoch_prom_t_bin_high_prob_pos_mask']`.
- For each epoch, build a **3D boolean/float volume** from its time-bin masks.
- Display that volume as a single 3D item (using Silx `ScalarField3D` or an equivalent item) in the same `SceneWindow` as the time-bin height maps.
- Keep this feature **optional**, controlled by a parameter on the viewer.

### Steps

- **Add configuration & state to viewer**
- Introduce a new parameter flag on `Epoch3DSceneTimeBinViewer` (e.g. `self.params.show_epoch_mask_stack_3d = False` by default).
- Add an attribute to hold the 3D stack item (e.g. `self.plots.epoch_mask_stack_item = None`) so we can manage its lifetime alongside the existing time-bin items.

- **Build 3D mask volume for the current epoch**
- In a new helper (e.g. `_build_epoch_mask_stack_volume(self, epoch_idx: int) -> Optional[np.ndarray]`):
- Access the per-epoch, per-time-bin masks from `self.plots_data.peak_contours['epoch_prom_t_bin_high_prob_pos_mask']`.
- Extract the slice for `epoch_idx` to get a 3D array of shape `(n_x_bins, n_y_bins, n_time_bins)` (or similar, depending on stored layout).
- Reorder axes to match Silx volume expectations (e.g. `(z, y, x)`), document the chosen ordering clearly in comments.
- Cast to `float32` (0/1) for use as a scalar field volume.

- **Create and position the 3D stack item**
- Implement `_create_epoch_mask_stack_item(self)` which:
- Calls `_build_epoch_mask_stack_volume` for `self.params.curr_epoch_idx`.
- If no data or flag is disabled, returns early.
- Creates a `ScalarField3D`-style item (either via `silx.gui.plot3d.SceneWindow.items.ScalarField3D` or another appropriate volume item, consistent with the Silx examples in [`Spike3D/LibrariesExamples/Silx/silx_examples/plot3dSceneWindow.py`](Spike3D/LibrariesExamples/Silx/silx_examples/plot3dSceneWindow.py)).
- Sets its data to the mask volume.
- Uses the same **XY extents** as the decoded surfaces (from `self.plots_data.x_min/x_max`, `self.plots_data.y_min/y_max`) so the stack aligns with `t_bin_idx=0` in world coordinates.
- Applies a **translation along Y** by `-k * self.plots_data.y_extent` (with `k≈1.0`), achieving the requested `y = -SIZE` style offset while staying data-driven.
- Optionally tweaks scale/isosurface thresholds so the stack is visually clear but not overwhelming (e.g. a single isosurface at 0.5 with semi-transparent color).
- Adds the item to `self.plots.scene_widget` and stores it in `self.plots.epoch_mask_stack_item`.

- **Clean up 3D stack item with epoch changes**
- Extend `_clear_time_bin_items` (or add a small companion method) to:
- Remove `self.plots.epoch_mask_stack_item` from the scene widget if present.
- Reset `self.plots.epoch_mask_stack_item = None`.
- Ensure this runs before creating new items in `on_epoch_changed` so we never leak visuals between epochs.

- **Integrate into epoch change lifecycle**
- In `on_epoch_changed`:
- After calling `_create_time_bin_items()` and `_add_contours_for_current_epoch()`, call `_create_epoch_mask_stack_item()` **only if** `self.params.show_epoch_mask_stack_3d` is `True` and the relevant mask data is available.
- Optionally provide a small public method (e.g. `setShowEpochMaskStack3D(self, enabled: bool)`) that toggles the flag and rebuilds the scene for the current epoch.

- **(Optional) Developer-facing notes / small example**
- Add a short docstring or comment block near `Epoch3DSceneTimeBinViewer` showing how to enable the 3D stack:
- Example: set `viewer.params.show_epoch_mask_stack_3d = True` after constructing the viewer, then call `on_epoch_changed(viewer.params.curr_epoch_idx)` or just rely on its existing initialization path.
- Briefly document the coordinate convention chosen for the volume (which axis is time vs X vs Y).