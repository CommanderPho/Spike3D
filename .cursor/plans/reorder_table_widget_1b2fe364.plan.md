---
name: Reorder table widget
overview: Swap two entries in `display()` so `self.table_widget` appears above `self.output_widget`.
todos:
  - id: swap-layout-order
    content: Swap table HBox and output_widget HBox in display() out_list.extend
    status: completed
isProject: false
---

# Reorder table above output area

## Change

In [`PhoDiba2023Paper.py`](h:/TEMP/Spike3DEnv_ExploreUpgrade/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py) `display()` (~3743–3756), swap the two children passed to `out_list.extend`:

**Current order:** `output_widget` HBox → table/`hover_posterior` HBox  
**Desired order:** table/`hover_posterior` HBox → `output_widget` HBox

```python
out_list.extend([
    widgets.HBox(
        [v for v in (self.table_widget, self.hover_posterior_preview_figure_widget) if v is not None],
        layout=widgets.Layout(height='300px', width='100%', display='flex', justify_content='space-between'),
    ),
    widgets.HBox([self.output_widget],
        layout=widgets.Layout(height='300px', width='100%'),
    ),
])
```

No other files or logic changes.