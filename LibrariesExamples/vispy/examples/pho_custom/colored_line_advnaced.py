# -*- coding: utf-8 -*-
# vispy: gallery 10:120:10
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
Changing Line Colors
====================
"""
from nptyping import NDArray
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any

from vispy import app, scene
from vispy.color import Colormap
from vispy.visuals.transforms import STTransform

def _example_with_colormap(custom_cmap: Colormap):
    # vertex positions of data to draw
    # N = 200
    N: int = 20
    pos = np.zeros((N, 2), dtype=np.float32)
    pos[:, 0] = np.linspace(10, 390, N)
    pos[:, 1] = np.random.normal(size=N, scale=20, loc=0) + (0.5 * pos[:, 0])

    # Sample the colormap at each vertex (0 to 1 along the line)
    t = np.linspace(0.0, 1.0, N)
    vertex_colors = np.array(custom_cmap.map(t), dtype=np.float32)
    print(f'\tvertex_colors: {vertex_colors}, np.shape(vertex_colors): {np.shape(vertex_colors)}')
    print(f'\tnp.shape(vertex_colors[0]): {np.shape(vertex_colors[0])}')
    print(f'\tnp.shape(vertex_colors[1]): {np.shape(vertex_colors[1])}')
    
    data_dict = dict(pos=pos, N=N, t=t, vertex_colors=vertex_colors)
    
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)

    line = scene.Line(pos=pos, color=vertex_colors, method='gl')
    line.transform = STTransform(translate=[0, 140])
    line.parent = canvas.central_widget

    text = scene.Text('custom: red→yellow→green→teal→blue→purple', bold=True, font_size=24, color='w', pos=(200, 40), parent=canvas.central_widget)
    return canvas, line, text, data_dict


# def on_timer(event):
#     global colormaps, line, text, pos
#     color = next(colormaps)
#     line.set_data(pos=pos, color=color)
#     text.text = color

# timer = app.Timer(.5, connect=on_timer, start=True)


if __name__ == '__main__':
    # Custom colormap transitioning: red -> yellow -> green -> teal -> blue -> purple
    # # ==================================================================================================================================================================================================================================================================================== #
    # # From list of colors Example                                                                                                                                                                                                                                                          #
    # # ==================================================================================================================================================================================================================================================================================== #
    # custom_cmap: Colormap = Colormap(colors=['red', 'yellow', 'green', 'teal', 'blue', 'purple'], controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # canvas, line, text, data_dict = _example_with_colormap(custom_cmap=custom_cmap)

    # ==================================================================================================================================================================================================================================================================================== #
    # From NDArray Example                                                                                                                                                                                                                                                                 #
    # ==================================================================================================================================================================================================================================================================================== #
    time_bin_colors = np.array([[0.9, 0.18, 0.18, 0.9],
    [0.9, 0.9, 0.18, 0.9],
    [0.18, 0.9 ,0.18, 0.9],
    [0.18, 0.9, 0.9, 0.9],
    [0.18, 0.18, 0.9, 0.9],
    [0.9, 0.18, 0.9, 0.9],
    ])

    n_time_bin_colors: int = np.shape(time_bin_colors)[0] #  np.shape(time_bin_colors): (6, 4)
    print(f'\ttime_bin_colors: {time_bin_colors}')
    colors_from_NDArray: List[NDArray] = [time_bin_colors[i][:3] for i in np.arange(n_time_bin_colors)]
    print(f'\tcolors_from_NDArray: {colors_from_NDArray}')
    custom_cmap = Colormap(colors=colors_from_NDArray) # , controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print(f'\tcustom_cmap: {custom_cmap}')
    canvas, line, text, data_dict = _example_with_colormap(custom_cmap=custom_cmap)

    canvas.app.run()
