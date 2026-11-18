---
name: ColorPickerDialog QColorDialog compatibility
overview: Make ColorPickerDialog API-compatible with QColorDialog by adding the signals and methods that ColorButton (and GradientEditorItem) expect, so it works as a drop-in without changing the base class to QColorDialog.
todos: []
isProject: false
---

# ColorPickerDialog QColorDialog-compatible API

## Current gap

[ColorButton.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqtgraph\widgets\ColorButton.py) uses the dialog as if it were a `QColorDialog`:

- **Signals:** `currentColorChanged`, `colorSelected`, `rejected`
- **Methods:** `setCurrentColor(color)`, `open()`

[ColorPickerDialog](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqt_color_picker\colorPickerDialog.py) currently inherits from `QDialog` and has:

- `getColor()` (same as “current” color)
- No `currentColorChanged` or `colorSelected` (only a commented stub)
- No `setCurrentColor()`; Cancel uses `close()` instead of `reject()` (so `rejected` still fires when the dialog closes, but using `reject()` is clearer)

[ColorPickerWidget](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqt_color_picker\colorPickerWidget.py) already provides `colorChanged`, `setCurrentColor()`, and `getCurrentColor()`, so the dialog can forward these.

## Recommended approach: keep QDialog, add QColorDialog-compatible API

Keep `ColorPickerDialog` as a `QDialog` subclass and add the same signals and methods as `QColorDialog`. No changes required in ColorButton or GradientEditorItem.

### 1. Add signals and forward from the widget

In [colorPickerDialog.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqt_color_picker\colorPickerDialog.py):

- Import `pyqtSignal` from `PyQt5.QtCore`.
- Define class-level signals (same names and types as QColorDialog):
  - `currentColorChanged = pyqtSignal(QColor)` — emitted whenever the picker’s color changes.
  - `colorSelected = pyqtSignal(QColor)` — emitted when the user accepts (OK) with the chosen color.
- After creating `__colorPickerWidget` in `__initUi`, connect:
  - `self.__colorPickerWidget.colorChanged.connect(self.currentColorChanged.emit)`  
  so any change in the widget (sliders, editor, etc.) is exposed as `currentColorChanged`.
- In `accept()`: emit `colorSelected(self.getColor())`, then call `super().accept()`.

### 2. Add QColorDialog-style methods

- **setCurrentColor(self, color)**  
Normalize `color` (QColor or string) and call `self.__colorPickerWidget.setCurrentColor(...)`. This will update the widget and, via the connection above, emit `currentColorChanged` (consistent with QColorDialog).
- **currentColor(self) -> QColor**  
Return `self.__colorPickerWidget.getCurrentColor()`. Keeps the same behavior as `getColor()` but matches QColorDialog’s name; `getColor()` can remain as an alias.

### 3. Cancel button and rejected

- Change Cancel from `cancelBtn.clicked.connect(self.close)` to `cancelBtn.clicked.connect(self.reject)`.  
`QDialog.rejected` is emitted when the dialog is rejected, so ColorButton’s `colorDialog.rejected.connect(self.colorRejected)` will continue to work and the intent is clearer.

### 4. Optional: static getColor() (QColorDialog-style)

If you want call sites that use `QColorDialog.getColor(initial)` to be able to use `ColorPickerDialog.getColor(initial)` as well, add a static method:

```python
@staticmethod
def getColor(initial=QColor(255,255,255), parent=None, title='Pick the color'):
    dlg = ColorPickerDialog(color=initial, parent=parent)
    if title:
        dlg.setWindowTitle(title)
    return dlg.getColor() if dlg.exec_() == QDialog.Accepted else QColor()
```

Note: the instance method `getColor(self)` already exists; the static method would live alongside it (same name, different signature). Callers using the static form would use `ColorPickerDialog.getColor(...)`. Omit this if you do not need static usage.

## Alternative: subclass QColorDialog

Making `ColorPickerDialog` inherit from `QColorDialog` and use `ColorPickerWidget` as the main content would require replacing or hiding the native dialog content and wiring our widget to the base signals/methods. That is more invasive and platform- and Qt-version-sensitive. Recommend only if you need `isinstance(dlg, QColorDialog)` or strict substitution everywhere `QColorDialog` is used.

## Summary of code changes


| Location                                                                                                                                                                   | Change                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [colorPickerDialog.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqt_color_picker\colorPickerDialog.py) | Import `pyqtSignal`; add `currentColorChanged` and `colorSelected`; connect widget `colorChanged` → `currentColorChanged`; in `accept()` emit `colorSelected` then `super().accept()`; add `setCurrentColor()` and `currentColor()`; connect Cancel to `reject()`. |
| [ColorButton.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\External\pyqtgraph\widgets\ColorButton.py)             | No changes required.                                                                                                                                                                                                                                               |


After these updates, ColorPickerDialog will be a drop-in for the way ColorButton (and any code that uses the same QColorDialog API) expects the dialog to behave, without switching the base class to QColorDialog.