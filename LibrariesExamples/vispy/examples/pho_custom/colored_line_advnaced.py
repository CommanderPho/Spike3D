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



def generate_loop_de_loop_line(N: int = 200, x_start: float = 10.0, x_end: float = 390.0, slope: float = -0.6, loop_center_frac: float = 0.5, loop_width: int = 80, loop_radius: float = 40.0, noise_scale: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D line that trends downward, performs a loop-de-loop,
    and then continues downward.

    Returns
    -------
    pos : (N, 2) float32 ndarray
        Vertex positions
    t : (N,) float32 ndarray
        Normalized [0, 1] parameter (useful for colormaps)

    Usage:

        pos, t = generate_loop_de_loop_line(N=300, slope=-0.8, loop_center_frac=0.55, loop_width=100, loop_radius=50.0, noise_scale=2.0)

    """
    pos = np.zeros((N, 2), dtype=np.float32)

    # Base x trajectory
    pos[:, 0] = np.linspace(x_start, x_end, N)

    # Base downward linear trend
    pos[:, 1] = slope * pos[:, 0]

    # Loop placement
    loop_center_idx = int(loop_center_frac * (N - 1))
    loop_width = min(loop_width, N)
    half_width = loop_width // 2

    loop_start = max(0, loop_center_idx - half_width)
    loop_end = min(N, loop_start + loop_width)
    loop_width = loop_end - loop_start  # recompute in case clipped

    # Parametric loop
    theta = np.linspace(0.0, 2.0 * np.pi, loop_width, endpoint=True)

    pos[loop_start:loop_end, 0] += loop_radius * np.cos(theta)
    pos[loop_start:loop_end, 1] += loop_radius * np.sin(theta)

    # Optional noise
    if noise_scale > 0.0:
        pos[:, 1] += np.random.normal(scale=noise_scale, size=N)

    # Colormap parameter
    t = np.linspace(0.0, 1.0, N, dtype=np.float32)

    return pos, t




def _example_with_colormap(custom_cmap: Colormap):
    # vertex positions of data to draw
    N: int = 200

    # N: int = 20
    # pos = np.zeros((N, 2), dtype=np.float32)
    # pos[:, 0] = np.linspace(10, 390, N)
    # pos[:, 1] = np.random.normal(size=N, scale=20, loc=0) + (0.5 * pos[:, 0])

    # # Sample the colormap at each vertex (0 to 1 along the line)
    # t = np.linspace(0.0, 1.0, N)

    pos, t = generate_loop_de_loop_line(N=N, slope=-0.8, loop_center_frac=0.55, loop_width=100, loop_radius=50.0, noise_scale=2.0)


    vertex_colors = np.array(custom_cmap.map(t), dtype=np.float32)
    print(f'\tvertex_colors: {vertex_colors}, np.shape(vertex_colors): {np.shape(vertex_colors)}')
    print(f'\tnp.shape(vertex_colors[0]): {np.shape(vertex_colors[0])}')
    print(f'\tnp.shape(vertex_colors[1]): {np.shape(vertex_colors[1])}')
    
    data_dict = dict(pos=pos, N=N, t=t, vertex_colors=vertex_colors)
    
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)

    line = scene.Line(pos=pos, color=vertex_colors, method='gl')
    # line.transform = STTransform(translate=[0, 140])
    line.parent = canvas.central_widget

    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)

    line = scene.Line(pos=pos, color=vertex_colors, method='gl')
    line.parent = view.scene

    # Auto-center and scale to data
    view.camera.set_range(
        x=(pos[:, 0].min(), pos[:, 0].max()),
        y=(pos[:, 1].min(), pos[:, 1].max()),
    )

    text = scene.Text('custom: red→yellow→green→teal→blue→purple', bold=True, font_size=24, color='w', pos=(200, 40), parent=canvas.central_widget)
    return canvas, line, text, data_dict



def _example_with_heading_color():
    """Example: draw a path colored by heading (0°=red, ROYGBIV, 359°=violet). Run with: python -c \"from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import example_heading_rainbow_line; example_heading_rainbow_line()\"."""
    from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import create_heading_rainbow_line

    # vertex positions of data to draw
    N: int = 300
    pos, t = generate_loop_de_loop_line(N=N, slope=-0.8, loop_center_frac=0.55, loop_width=100, loop_radius=50.0, noise_scale=0.1)

    ## pos
    data_dict = dict(pos=pos, N=N, t=t)
    
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)

    scene_parent = view.scene
    if scene_parent is not None:
        line = create_heading_rainbow_line(pos=pos, parent=scene_parent, line_width=3.0, order=10)
        line.set_gl_state('translucent', depth_test=False)


    # Auto-center and scale to data
    view.camera.set_range(
        x=(pos[:, 0].min(), pos[:, 0].max()),
        y=(pos[:, 1].min(), pos[:, 1].max()),
    )

    text = scene.Text('custom: _example_with_heading_color', bold=True, font_size=24, color='w', pos=(200, 40), parent=canvas.central_widget)
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
    # time_bin_colors = np.array([[0.9, 0.18, 0.18, 0.9],
    # [0.9, 0.9, 0.18, 0.9],
    # [0.18, 0.9 ,0.18, 0.9],
    # [0.18, 0.9, 0.9, 0.9],
    # [0.18, 0.18, 0.9, 0.9],
    # [0.9, 0.18, 0.9, 0.9],
    # ])
    # n_time_bin_colors: int = np.shape(time_bin_colors)[0] #  np.shape(time_bin_colors): (6, 4)
    # print(f'\ttime_bin_colors: {time_bin_colors}')
    # colors_from_NDArray: List[NDArray] = [time_bin_colors[i][:3] for i in np.arange(n_time_bin_colors)]
    # print(f'\tcolors_from_NDArray: {colors_from_NDArray}')
    # custom_cmap = Colormap(colors=colors_from_NDArray) # , controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # print(f'\tcustom_cmap: {custom_cmap}')
    # canvas, line, text, data_dict = _example_with_colormap(custom_cmap=custom_cmap)

    canvas, line, text, data_dict = _example_with_heading_color()


    canvas.app.run()
