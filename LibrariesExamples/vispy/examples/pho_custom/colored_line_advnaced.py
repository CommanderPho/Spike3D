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
import vispy.plot as vp
from vispy.color import Colormap
from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import VispyHelpers


# def add_line_axes(view, pos):
#     title = scene.Label("Plot Title", color='white')
#     title.height_max = 40
#     grid.add_widget(title, row=0, col=0, col_span=2)

#     yaxis = scene.AxisWidget(orientation='left',
#                             axis_label='Y Axis',
#                             axis_font_size=12,
#                             axis_label_margin=50,
#                             tick_label_margin=5)
#     yaxis.width_max = 80
#     grid.add_widget(yaxis, row=1, col=0)

#     xaxis = scene.AxisWidget(orientation='bottom',
#                             axis_label='X Axis',
#                             axis_font_size=12,
#                             axis_label_margin=50,
#                             tick_label_margin=5)

#     xaxis.height_max = 80
#     grid.add_widget(xaxis, row=2, col=1)





def _example_with_colormap(custom_cmap: Colormap):
    # vertex positions of data to draw
    N: int = 200

    # N: int = 20
    # pos = np.zeros((N, 2), dtype=np.float32)
    # pos[:, 0] = np.linspace(10, 390, N)
    # pos[:, 1] = np.random.normal(size=N, scale=20, loc=0) + (0.5 * pos[:, 0])

    # # Sample the colormap at each vertex (0 to 1 along the line)
    # t = np.linspace(0.0, 1.0, N)

    pos, t = VispyHelpers.generate_loop_de_loop_line(N=N, slope=-0.8, loop_center_frac=0.55, loop_width=100, loop_radius=50.0, noise_scale=2.0)


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
    from pyphoplacecellanalysis.Pho2D.vispy.position_heading_angle import CompassLegendItem
    
    # from pyphoplacecellanalysis.Pho2D.vispy.vispy_helpers import create_heading_rainbow_line

    # vertex positions of data to draw
    N: int = 300
    pos, t = VispyHelpers.generate_loop_de_loop_line(N=N, slope=-0.8, loop_center_frac=0.55, loop_width=100, loop_radius=50.0, noise_scale=0.0)

    ## pos
    data_dict = dict(pos=pos, N=N, t=t)
    
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 400), show=True)
    grid = canvas.central_widget.add_grid(spacing=10)

    # ---- Top: one big view spanning 3 columns ----
    main_view = grid.add_view(row=0, col=0, col_span=3, camera='panzoom')
    main_view.border_color = (1, 1, 1, 1)

    # ---- Bottom: three small views ----
    view_a = grid.add_view(row=1, col=0, camera='panzoom')
    view_b = grid.add_view(row=1, col=1, camera='panzoom')
    view_c = grid.add_view(row=1, col=2, camera='panzoom')

    for v in (view_a, view_b, view_c):
        v.border_color = (0.6, 0.6, 0.6, 1)


    # view = canvas.central_widget.add_view()
    # view.camera = scene.PanZoomCamera(aspect=1)

    scene_parent = main_view.scene
    if scene_parent is not None:
        line, data_dict = VispyHelpers.create_heading_rainbow_line(pos=pos, parent=scene_parent, line_width=1.0, order=10)
        line.set_gl_state('translucent', depth_test=False)


    # Auto-center and scale to data
    main_view.camera.set_range(
        x=(pos[:, 0].min(), pos[:, 0].max()),
        y=(pos[:, 1].min(), pos[:, 1].max()),
    )

    text = scene.Text('custom: _example_with_heading_color', bold=True, font_size=24, color='w', pos=(200, 40), parent=canvas.central_widget)

    ## has a canvas (SceneCanvas)
    fig = vp.Fig(size=(800, 400), show=False)
    plotwidget = fig[0, 0]

    ## Sub-views:
    headings_deg = data_dict.get('headings_deg', None)
    if headings_deg is not None:
        headings_deg_pos = VispyHelpers.build_line_pos(t, headings_deg)
        
        fig.title = "headings_deg_pos"
        plotwidget.plot(headings_deg_pos, title="headings_deg_pos")
        # plotwidget.colorbar(position="top", cmap="autumn")
        # legend_pos, legend_headings_deg, legend_t = VispyHelpers.generate_angular_shading_legend(x_center=(0, 0), radius=20)
        scene_parent = view_a.scene
        if scene_parent is not None:
            # legend_line, legend_data_dict = VispyHelpers.create_heading_rainbow_line(pos=legend_pos, headings_deg=legend_headings_deg, parent=scene_parent, line_width=4, order=11)
            # legend_line.set_gl_state('translucent', depth_test=False)
            
            legend_out = CompassLegendItem(view=view_a)
            legend_pos = legend_out._data_dict.get('pos', None)        
            
            # scene.visuals.Line(pos=headings_deg_pos, parent=view_a.scene)
            VispyHelpers.set_view_camera(view_a, pos=legend_pos)
            # print(f'legend_data_dict: {legend_data_dict}')


    pos = VispyHelpers.build_line_pos(t, pos[:, 0])
    plotwidget = fig[0, 1]
    plotwidget.plot(pos, title="x")

    # scene.visuals.Line(pos=pos, parent=view_b.scene)
    # set_view_camera(view_b, pos=pos)

    pos = VispyHelpers.build_line_pos(t, pos[:, 1])
    plotwidget = fig[0, 2]
    plotwidget.plot(pos, title="y")    

    # scene.visuals.Line(pos=pos, parent=view_c.scene)
    # set_view_camera(view_c, pos=pos)

    return canvas, line, text, data_dict, fig


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

    canvas, line, text, data_dict, fig = _example_with_heading_color()
    if fig is not None:
        fig.show()

    canvas.app.run()
