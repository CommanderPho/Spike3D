#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
User Interface Rendering and interactivity helpers
"""

import pyvista as pv

def print_controls_helper_text():
    controls_helper_text_strings = ['[f] - Focus and zoom in on the last clicked point',
        'shift+click - Drag to pan the rendering scene',
        'ctrl+click - Rotate the scene in 2D']
    controls_helper_text = '\n'.join(controls_helper_text_strings)
    print(controls_helper_text)
    return controls_helper_text
    
def _customize_default_slider_gui_style(my_theme):
    my_theme.slider_styles.modern.slider_length = 0.01
    my_theme.slider_styles.modern.slider_width = 0.05 # the height of the slider releative to the bar
    my_theme.slider_styles.modern.slider_color = (0.13, 0.14, 0.15) # (0.43, 0.44, 0.45)
    my_theme.slider_styles.modern.tube_width = 0.02
    my_theme.slider_styles.modern.tube_color = (0.69, 0.40, 0.109)
    my_theme.slider_styles.modern.cap_opacity = 0.0
    my_theme.slider_styles.modern.cap_length = 0.01
    my_theme.slider_styles.modern.cap_width = 0.02
    return my_theme

def customize_default_pyvista_theme():
    my_theme = pv.themes.DefaultTheme()
    my_theme = _customize_default_slider_gui_style(my_theme)
    my_theme.window_size = [1920, 1080]
    ## Apply the theme as the active pyvista theme
    print('Applying custom Pyvista theme.')
    pv.global_theme.load_theme(my_theme)
    # pv.global_theme.nan_opacity=0.0
    print('done.')
    
    

class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified. 
    Usage:
        # Make a separate callback for each widget
        callback = SetVisibilityCallback(actor)
    """
    def __init__(self, actor):
        self.actor = actor

    def __call__(self, state):
        self.actor.SetVisibility(state)
        
        