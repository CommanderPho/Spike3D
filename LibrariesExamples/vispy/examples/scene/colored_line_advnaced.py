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
import numpy as np

from vispy import app, scene
from vispy.color import Colormap
from vispy.visuals.transforms import STTransform

# Custom colormap transitioning: red -> yellow -> green -> teal -> blue -> purple
custom_cmap = Colormap(colors=['red', 'yellow', 'green', 'teal', 'blue', 'purple'], controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# vertex positions of data to draw
# N = 200
N = 20
pos = np.zeros((N, 2), dtype=np.float32)
pos[:, 0] = np.linspace(10, 390, N)
pos[:, 1] = np.random.normal(size=N, scale=20, loc=0) + (0.5 * pos[:, 0])

# Sample the colormap at each vertex (0 to 1 along the line)
t = np.linspace(0.0, 1.0, N)
vertex_colors = np.array(custom_cmap.map(t), dtype=np.float32)

canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)

line = scene.Line(pos=pos, color=vertex_colors, method='gl')
line.transform = STTransform(translate=[0, 140])
line.parent = canvas.central_widget

text = scene.Text('custom: red→yellow→green→teal→blue→purple', bold=True, font_size=24, color='w', pos=(200, 40), parent=canvas.central_widget)


# def on_timer(event):
#     global colormaps, line, text, pos
#     color = next(colormaps)
#     line.set_data(pos=pos, color=color)
#     text.text = color

# timer = app.Timer(.5, connect=on_timer, start=True)


if __name__ == '__main__':
    canvas.app.run()
