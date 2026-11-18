---
name: Fix header context menu
overview: The column-visibility context menu is wired to `QTreeWidget.header()` only. On many Qt builds, `QHeaderView` receives mouse events on its internal **viewport**, so the header widget never emits `customContextMenuRequested` and no menu appears. Fix by applying the same policy + signal to `header.viewport()` and mapping coordinates correctly (with a fallback if viewport is missing).
todos:
  - id: viewport-policy-signal
    content: Set CustomContextMenu on header.viewport(), connect customContextMenuRequested from viewport; fix mapToGlobal to use viewport/sender
    status: completed
  - id: optional-enum-helper
    content: Add compact Qt5/Qt6-safe ContextMenuPolicy helper if getattr path is unclear
    status: completed
  - id: verify-manually
    content: Verify right-click on header shows menu at cursor
    status: completed
isProject: false
---

# Fix VispySceneTreeWidget header context menu

## Root cause

`[VispySceneTreeWidget._init_ui](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)` sets `CustomContextMenu` and connects `customContextMenuRequested` only on `self.tree.header()` (lines 610–624). `QHeaderView` is built on top of an internal **viewport** (via `QAbstractScrollArea`); right-clicks on the visible header area often hit the **viewport** child, which keeps the default context-menu policy and never emits the parent header’s signal—so **no handler runs** and **no menu appears**.

This differs from “works on QTableView” setups only in that tree/header embedding and platform/style can make the mismatch more visible; the robust pattern is to handle the viewport explicitly.

## Implementation (single file)

**File:** `[pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/Pho2D/vispy/vispy_helpers.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\Pho2D\vispy\vispy_helpers.py)`

1. **Introduce a small helper** (e.g. `_context_menu_policy_custom()`) that returns `Qt.ContextMenuPolicy.CustomContextMenu` on Qt6 and `Qt.CustomContextMenu` on Qt5, so `setContextMenuPolicy` always receives the correct enum (optional but avoids subtle getattr issues across qtpy bindings).
2. **After obtaining `header = self.tree.header()`**, also get `viewport = header.viewport()` (if not `None`):
  - Call `setContextMenuPolicy(custom_policy)` on **both** `header` and `viewport` (viewport is the important one for receiving events).
  - Connect `customContextMenuRequested` **from the viewport** to `_on_tree_header_context_menu` (or a thin wrapper that knows the sender).
  - Optionally keep the header connection as well for redundancy, or **only** connect the viewport to avoid duplicate menus if both ever fired (prefer **viewport-only** connection once verified).
3. **Update `_on_tree_header_context_menu(self, pos)`** so the global position passed to `QMenu.exec` / `exec_` uses the **widget that emitted the signal**:
  - If the slot is connected from `viewport`, use `header.viewport().mapToGlobal(pos)` (or `self.sender().mapToGlobal(pos)` with a cast) instead of `header.mapToGlobal(pos)`.
  - Wrong mapping can place the menu off-screen or at (0,0); it does not explain “no menu,” but fix it together with the viewport connection.
4. **Manual check:** Run the app, right-click directly on a **column title** in the scene tree; the column visibility menu should appear at the cursor.

## Fallback (only if viewport fix is insufficient)

If some platform still fails, add a `QEvent` filter on the header (or viewport) for `QEvent.ContextMenu` and show the same `QMenu` at `event.globalPos()`. This should not be necessary if step 2 is done correctly.

## References in repo

- Same **header-only** pattern exists in `[StackedDynamicTablesWidget.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Widgets\Testing\StackedDynamicTablesWidget.py)` (`QTableView.horizontalHeader()`); `QTreeWidget` + `QHeaderView.viewport()` is the more reliable target for trees.

