---
title: Jupyter Notebook Pandas Dataframe Display
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/53a0583d-0248-471d-b1fb-a9ce8bb1fbfd
url: https://phohale.sourcegraph.app/deepsearch/306
created: 2025-12-15T09:42:16Z
updated: 2026-01-15T10:15:29Z
---

# github.com/CommanderPho/Spike3D github.com/CommanderPho/pyPhoCoreHelpers github.com/CommanderPho/pyPhoPlaceCellAnalysis github.com/diba-lab/NeuroPy Pandas Dataframe display helper for Jupyter notebooks that permits scrollability and customizable number of rows.

The **pyPhoCoreHelpers** library provides a Pandas DataFrame display helper for Jupyter notebooks called `render_scrollable_colored_table_from_dataframe` in [print_helpers.py](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoCoreHelpers/-/blob/src/pyphocorehelpers/print_helpers.py?L1701-L1851).

## Two Usage Methods

### 1. **Direct Function Call** (Recommended)

```python
from pyphocorehelpers.print_helpers import render_scrollable_colored_table_from_dataframe

# Basic usage
render_scrollable_colored_table_from_dataframe(df)

# With customization
render_scrollable_colored_table_from_dataframe(
    df=my_dataframe,
    cmap_name='plasma',           # matplotlib colormap for cell colors
    max_height=500,               # scrollable container height in px
    width='80%',                  # table width
    max_rows_to_render_for_performance=100,  # truncate if more rows
    is_dark_mode=True             # theme
)
```

### 2. **IPython Magic Command** (Cell-wide)

```python
from pyphocorehelpers.ipython_helpers import CustomFormatterMagics

# Register the magic once
get_ipython().register_magics(CustomFormatterMagics)

# Then use in any cell
%%scrollable_colored_table
df1  # All DataFrames in this cell will be displayed with scrolling
df2
df3
```

## Key Features

The helper provides ([source](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoCoreHelpers/-/blob/src/pyphocorehelpers/print_helpers.py?L1701-L1851)):

1. **Scrollability**: Fixed-height container (default 400px) with vertical scrolling
2. **Customizable Row Display**: 
   - `max_rows_to_render_for_performance` parameter (default: 100)
   - Uses [`ellided_dataframe`](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoCoreHelpers/-/blob/src/pyphocorehelpers/print_helpers.py?L1639-L1668) to intelligently truncate with ellipsis rows
   - Shows footer with total rows/columns and truncation info
3. **Color Coding**: Optional matplotlib colormap-based cell coloring with automatic contrast-aware text colors
4. **Scroll Indicators**: Green gradient shadow when content exceeds visible area
5. **Dark/Light Mode**: Automatic theming support

## Additional Helper

There's also a [dataframe_show_more_button](https://phohale.sourcegraph.app/github.com/CommanderPho/pyPhoCoreHelpers/-/blob/src/pyphocorehelpers/preferences_helpers.py?L145-L244) function that adds an interactive "show more" button with a customizable row input field for dynamically expanding the display.

The diagram above illustrates the complete workflow and features of both approaches.

## Sources

- [repo:CommanderPho/Spike3D OR repo:CommanderPho/pyPhoCoreHelpers OR repo:CommanderPho/pyPhoPlaceCellAnalysis OR repo:diba-lab/NeuroPy pandas dataframe display jupyter scrollable rows](/search?q=patterntype:nls%20repo:CommanderPho%2FSpike3D%20OR%20repo:CommanderPho%2FpyPhoCoreHelpers%20OR%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:diba-lab%2FNeuroPy%20pandas%20dataframe%20display%20jupyter%20scrollable%20rows)
- [repo:CommanderPho/Spike3D OR repo:CommanderPho/pyPhoCoreHelpers OR repo:CommanderPho/pyPhoPlaceCellAnalysis OR repo:diba-lab/NeuroPy set_option display max_rows](/search?q=patterntype:nls%20repo:CommanderPho%2FSpike3D%20OR%20repo:CommanderPho%2FpyPhoCoreHelpers%20OR%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:diba-lab%2FNeuroPy%20set_option%20display%20max_rows)
- [(repo:CommanderPho/Spike3D OR repo:CommanderPho/pyPhoCoreHelpers OR repo:CommanderPho/pyPhoPlaceCellAnalysis OR repo:diba-lab/NeuroPy) pd.set_option](/search?q=patterntype:keyword%20%28repo:CommanderPho%2FSpike3D%20OR%20repo:CommanderPho%2FpyPhoCoreHelpers%20OR%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20OR%20repo:diba-lab%2FNeuroPy%29%20pd.set_option)
- [repo:CommanderPho/pyPhoCoreHelpers display helper scrollable html ipython](/search?q=patterntype:nls%20repo:CommanderPho%2FpyPhoCoreHelpers%20display%20helper%20scrollable%20html%20ipython)
- [repo:CommanderPho/pyPhoCoreHelpers display HTML scrollable](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoCoreHelpers%20display%20HTML%20scrollable)
- [repo:CommanderPho/pyPhoCoreHelpers to_html](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoCoreHelpers%20to_html)
- [repo:CommanderPho/pyPhoPlaceCellAnalysis display HTML](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoPlaceCellAnalysis%20display%20HTML)
- [github.com/CommanderPho/pyPhoCoreHelpers/src/pyphocorehelpers/print_helpers.py@066d6f3?L1634-1900](/github.com/CommanderPho/pyPhoCoreHelpers@066d6f3fe679ab0683169e3be844257d24301bf3/-/blob/src/pyphocorehelpers/print_helpers.py?L1634-1900)
- [github.com/CommanderPho/pyPhoCoreHelpers/src/pyphocorehelpers/ipython_helpers.py@066d6f3](/github.com/CommanderPho/pyPhoCoreHelpers@066d6f3fe679ab0683169e3be844257d24301bf3/-/blob/src/pyphocorehelpers/ipython_helpers.py)
- [repo:CommanderPho/pyPhoCoreHelpers %%scrollable_colored_table](/search?q=patterntype:keyword%20repo:CommanderPho%2FpyPhoCoreHelpers%20%25%25scrollable_colored_table)
- [repo:CommanderPho render_scrollable_colored_table_from_dataframe](/search?q=patterntype:keyword%20repo:CommanderPho%20render_scrollable_colored_table_from_dataframe)
- [github.com/CommanderPho/pyPhoCoreHelpers/src/pyphocorehelpers/preferences_helpers.py@066d6f3?L140-320](/github.com/CommanderPho/pyPhoCoreHelpers@066d6f3fe679ab0683169e3be844257d24301bf3/-/blob/src/pyphocorehelpers/preferences_helpers.py?L140-320)

## Suggested Follow-ups

- How does the color coding in `render_scrollable_colored_table_from_dataframe` handle different data types (e.g., strings, dates) in the DataFrame?
- Can you provide a more detailed example of using the `dataframe_show_more_button` helper function and how it interacts with the scrollable table?

