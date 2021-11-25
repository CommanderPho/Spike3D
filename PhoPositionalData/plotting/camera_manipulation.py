#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
Camera control functions
"""
import sys
import pyvista as pv
import pyvistaqt as pvqt


# Camera Updating Functions:
def apply_camera_view(p, desired_camera_position_matrix):
    # desired_camera_position_matrix is something like [(-53.34019399163205, -162.05112717599195, 141.89039137933437), (-2.783721923828125, 13.290399551391602, 1.0999999046325686), (-0.23261915078042855, 0.6488198771982244, 0.7245143874642332)]
    previous_camera_position = p.camera_position
    print('previous_camera_position: {}'.format(previous_camera_position))
    p.camera_position = desired_camera_position_matrix
    # p.update()
    p.render()
    return previous_camera_position

def apply_close_perspective_camera_view(p):
    close_perspective_camera_position = [(-53.34019399163205, -162.05112717599195, 141.89039137933437), (-2.783721923828125, 13.290399551391602, 1.0999999046325686), (-0.23261915078042855, 0.6488198771982244, 0.7245143874642332)]
    return apply_camera_view(p, close_perspective_camera_position) 
    
def apply_close_overhead_zoomed_camera_view(p):
    desired_overhead_zoomed_camera_position = [(-2.783721923828125, 13.290399551391602, 546.7586522921252), (-2.783721923828125, 13.290399551391602, 1.0999999046325684), (0.0, 1.0, 0.0)]
    return apply_camera_view(p, desired_overhead_zoomed_camera_position)
