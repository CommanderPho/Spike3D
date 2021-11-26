#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
User Interface Rendering and interactivity helpers
"""
import numpy as np
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
    
    
###########################    
## Playback Timestamp Slider Adjustments Programmatically Methods:

class MutuallyExclusiveRadioButtonGroup:
    """Enforces the constraint that exactly one is active at a time
    Usage:

        
        # Make a separate callback for each widget
        callback = SetVisibilityCallback(actor)
        callbacks
        
        mutually_exclusive_radiobutton_group = MutuallyExclusiveRadioButtonGroup(len(callbacks), active_element_idx=0, on_element_state_changed_callbacks=callbacks)
        mutually_exclusive_radiobutton_group[5] = True # activate element 5
    """
    
    # def __init__(self, num_elements, active_element_idx=0, on_activate_element_callbacks=None, on_deactivate_element_callbacks=None):    
    #     if on_activate_element_callbacks is not None:
    #         self._on_activate_element_callbacks = on_activate_element_callbacks
            
    #     if on_deactivate_element_callbacks is not None:
    #         self._on_deactivate_element_callbacks = on_deactivate_element_callbacks
        
    def __init__(self, num_elements, active_element_idx=0, on_element_state_changed_callbacks=None):
        # on_element_state_changed_callbacks: a list of callbacks for each of the num_elements that accept one boolean argument indicating the udpated state of that element. Called upon change.
        self._on_element_state_changed_callbacks = on_element_state_changed_callbacks
        
                
        # Allocate the boolean array that indicates which element is active
        self._is_element_active = np.full([num_elements,], False)
        
        self._active_element_idx = active_element_idx        
        self._is_element_active[self._active_element_idx] = True # set one element to true                
        self.perform_callback(active_element_idx)
        
    def __getitem__(self, index):
        return self._is_element_active[index]
    
    def __setitem__(self, index, new_value):
        if new_value:
            # normal case, new active item selected
            self.set_element_active(index)
        else:
            # unusual case, an item is set to be inactivated
            if index == self._active_element_idx:
                # The index to be deactivated is the active one, so change the active index to 0
                self.set_element_active() # set the default (0) element active since the active one is being inactivated
            else:
                # only when the index to be deactivated is the active one will anything actually happen
                pass
        
        
    def set_element_active(self, proposed_active_element_idx=0):
        if self._active_element_idx == proposed_active_element_idx:
            # element unchanged
            pass
        else:
            # element changing
            prev_active_element_idx = self._active_element_idx
            # Inactivate the old element:
            self._is_element_active[prev_active_element_idx] = False # set prev element to false
            self.perform_callback(prev_active_element_idx)
            # Activate the new element:
            self._active_element_idx = proposed_active_element_idx
            self._is_element_active[self._active_element_idx] = True # set new active element to true
            self.perform_callback(self._active_element_idx)
                
                
    def perform_callback(self, callback_idx):
        if self._on_element_state_changed_callbacks is not None:
                curr_callback = self._on_element_state_changed_callbacks[callback_idx]
                curr_callback(self._is_element_active[callback_idx]) # pass the new state to the callback

    @property
    def active_element_idx(self):
        return self._active_element_idx

    @active_element_idx.setter
    def active_element_idx(self, proposed_active_element_idx):
        """active_element_index ensures consistency on set"""
        self.set_element_active(proposed_active_element_idx)

    @property
    def is_element_active(self):
        """
        Returns:
            [Bool]: [Returns a boolean array indicating where each element is active]
        """
        return self._is_element_active



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
        
        
        
def add_placemap_toggle_checkboxes(p, placemap_actors, colors, widget_check_states=False, widget_size=20, widget_start_pos=12, widget_border_size=3):
    # """ Adds a list of toggle checkboxes to turn on and off each placemap"""
    if type(widget_check_states)==bool:
        widget_check_states = np.full([len(placemap_actors),], widget_check_states)
        
    curr_start_pos = widget_start_pos
    
    visibility_callbacks = list()
    checkboxWidgetActors = list()
    
    for i, an_actor in enumerate(placemap_actors):
        # Make a separate callback for each widget
        callback = SetVisibilityCallback(an_actor)
        callback(widget_check_states[i]) # perform the callback to update the initial visibility based on the correct state for this object
        visibility_callbacks.append(callback)
        curr_widget_actor = p.add_checkbox_button_widget(callback, value=widget_check_states[i],
                position=(5.0, curr_start_pos), size=widget_size,
                border_size=widget_border_size,
                color_on=colors[:,i],
                color_off='grey',
                background_color=colors[:,i] # background_color is used for the border
            ) 
        checkboxWidgetActors.append(curr_widget_actor)
        # compute updated start position
        curr_start_pos = curr_start_pos + widget_size + (widget_size // 10)
    return checkboxWidgetActors, visibility_callbacks



