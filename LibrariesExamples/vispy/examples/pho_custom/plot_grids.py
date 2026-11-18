# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
"""
Plot data with different styles in a grid of subplots
=====================================================

Renders n_fixed_columns subplots per row and computes n_rows from the total
number of plots.
"""

import math
import numpy as np

from vispy import plot as vp


def make_grid_figure(n_plots: int, n_fixed_columns: int = 6, fig_size=(900, 600), show=False, grid_spacing=0):
    """Create a vispy Fig with a grid of subplots: n_fixed_columns columns and n_rows rows."""
    n_rows = max(1, math.ceil(n_plots / n_fixed_columns))
    fig = vp.Fig(size=fig_size, show=show)
    # fig._grid.spacing = 0
    fig._grid.spacing = grid_spacing  # minimal padding between rows and columns (default is 6)
    fig.central_widget.padding = 0
    fig.central_widget.margin = 0
    return fig, n_rows, n_fixed_columns


n_fixed_columns = 6
n_plots = 14  # total number of subplots to render
fig, n_rows, n_cols = make_grid_figure(n_plots, n_fixed_columns=n_fixed_columns, fig_size=(900, 350), show=False, grid_spacing=1)

x = np.linspace(0, 10, 500)
colors = [(0.8, 0, 0, 1), (0.8, 0, 0.8, 1), (0, 0, 1.0, 1), (0, 0.7, 0, 1), (0.9, 0.5, 0, 1), (0, 0.8, 0.8, 1)]

for k in range(n_plots):
    row, col = k // n_fixed_columns, k % n_fixed_columns
    ax = fig[row, col]
    n_terms = k + 1
    y = np.zeros_like(x)
    L = 5
    for i in range(n_terms):
        n = i * 2 + 1
        y += (4. / np.pi) * (1. / n) * np.sin(n * np.pi * x / L)
    color = colors[k % len(colors)]
    ax.plot((x, y), width=2, color=color, title=f'n terms={n_terms}') # , xlabel='x', ylabel='y'
    grid = vp.visuals.GridLines(color=(0, 0, 0, 0.3))
    grid.set_gl_state('translucent')
    ax.view.add(grid)

if __name__ == '__main__':
    fig.show(run=True)
