#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
User Interface Rendering and interactivity helpers
"""
import numpy as np
import pyvista as pv

from PhoGui.PhoCustomVtkWidgets import PhoWidgetHelper

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
        mutually_exclusive_radiobutton_group = MutuallyExclusiveRadioButtonGroup(len(callbacks), active_element_idx=0, on_element_state_changed_callbacks=callbacks)
        mutually_exclusive_radiobutton_group[5] = True # activate element 5
    """
    def __init__(self, num_elements, active_element_idx=None, on_element_state_changed_callbacks=None, require_active_selection=False, is_debug=False):
        # on_element_state_changed_callbacks: a list of callbacks for each of the num_elements that accept one boolean argument indicating the udpated state of that element. Called upon change.
        # require_active_selection: if True, ensures that at least one active_element_idx is always selected, otherwise, allows active_element_idx to be None
        self.is_debug=is_debug
        self.num_elements = num_elements
        self.require_active_selection = require_active_selection
        self._on_element_state_changed_callbacks = on_element_state_changed_callbacks
     
        # Allocate the boolean array that indicates which element is active
        self._is_element_active = np.full([num_elements,], False)
        
        if self.require_active_selection:
            if active_element_idx is None:
                # if requires an active_selection, set self._active_element_idx to zero if it is None
                active_element_idx = 0
        self._active_element_idx = active_element_idx          
        
        if self._active_element_idx is not None:      
            self._is_element_active[self._active_element_idx] = True # set one element to true

        if self.is_debug:
            print('MutuallyExclusiveRadioButtonGroup(num_elements: {}, active_element_idx: {}, require_active_selection): Initialized'.format(num_elements, self._active_element_idx, require_active_selection))
            
        # Need to update all elements using the callback since they were all initialized to either False/True values:
        for i in np.arange(self.num_elements):
            self.perform_callback(i)

    def __getitem__(self, index):
        return self._is_element_active[index]
    
    def __setitem__(self, index, new_value):
        if new_value:
            # normal case, new active item selected
            self.set_element_active(index)
        else:
            # unusual case, an item is set to be inactivated
            if self.require_active_selection and (index == self._active_element_idx):
                # The index to be deactivated is the active one, so change the active index to 0
                self.set_element_active() # set the default (0) element active since the active one is being inactivated
            else:
                # only when the index to be deactivated is the active one will anything actually happen
                pass
        
    def __repr__(self) -> str:
        return f"<MutuallyExclusiveRadioButtonGroup: num_elements: {self.num_elements}, active_element_idx: {self.active_element_idx}>: {self.is_element_active}"

    def __str__(self) -> str:
        return f"<MutuallyExclusiveRadioButtonGroup: num_elements: {self.num_elements}, active_element_idx: {self.active_element_idx}>: {self.is_element_active}"

    
    def set_element_active(self, proposed_active_element_idx=0):
        if self._active_element_idx == proposed_active_element_idx:
            # element unchanged
            if self.is_debug:
                print('MutuallyExclusiveRadioButtonGroup.set_element_active({}): Active index unchanged'.format(proposed_active_element_idx))
            pass
        else:
            # element changing
            prev_active_element_idx = self._active_element_idx
            if self.is_debug:
                print('MutuallyExclusiveRadioButtonGroup.set_element_active({}): Active index changing from {} to {}'.format(proposed_active_element_idx, prev_active_element_idx, proposed_active_element_idx))
            # Inactivate the old element:
            if prev_active_element_idx is not None:
                self._is_element_active[prev_active_element_idx] = False # set prev element to false
                self.perform_callback(prev_active_element_idx)
            
            # Activate the new element:
            self._active_element_idx = proposed_active_element_idx
            if self._active_element_idx is not None:
                self._is_element_active[self._active_element_idx] = True # set new active element to true
                self.perform_callback(self._active_element_idx)
                
                
    def perform_callback(self, callback_idx):
        if self._on_element_state_changed_callbacks is not None:
                curr_callback = self._on_element_state_changed_callbacks[callback_idx]
                if self.is_debug:
                    print('MutuallyExclusiveRadioButtonGroup.perform_callback({}): {}'.format(callback_idx, self._is_element_active[callback_idx]))
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


class SetUICheckboxValueCallback:
    """Helper callback to keep a reference to a checkbox widget and allow it to be updated programmatically 
    Usage:
        # Make a separate callback for each widget
        callback = SetUICheckboxValueCallback(actor)
    """
    def __init__(self, checkbox_widget_actor, is_debug=False):
        self.is_debug=is_debug
        self.checkbox_widget_actor = checkbox_widget_actor

    def __call__(self, state=None):
        if state is None:
            # if no state provided, toggle the current state:
            prev_state = self.checkbox_widget_actor.GetRepresentation().GetState()
            state = not prev_state
        self.checkbox_widget_actor.GetRepresentation().SetState(state)
      

class OnUICheckboxChangedCallback:
    """Helper callback to keep a reference to both the checkbox widget and the index which it represents and allow it perform a custom callback when its value is changed interactively
    callback: a function that takes (widget_index: Int, state: Bool)
    Usage:
        def _update_mutually_exclusive_callback(widget_index, state):
            debug_console_widget.add_line_to_buffer('_update_mutually_exclusive_callback(widget[{}]): updated value {})'.format(widget_index, state))
            mutually_exclusive_radiobutton_group[widget_index] = state # set the mutually exclusive active element using the widget changed callback
        
        # add the function that responds to user initiated changes by clicking on the value
        checkbox_changed_callbacks = list()            
        for i, a_checkbox_widget_actor in enumerate(checkboxWidgetActors):
            curr_checkbox_changed_callback = OnUICheckboxChangedCallback(a_checkbox_widget_actor, i, _update_mutually_exclusive_callback, is_debug=is_debug)
            a_checkbox_widget_actor.AddObserver(pv._vtk.vtkCommand.StateChangedEvent, curr_checkbox_changed_callback)
            checkbox_changed_callbacks.append(curr_checkbox_changed_callback)
    """
    def __init__(self, checkbox_widget_actor, widget_index, on_update_callback, is_debug=False):
        self.is_debug=is_debug
        self.checkbox_widget_actor = checkbox_widget_actor
        self.widget_index = widget_index
        self.callback = on_update_callback
    
    def __call__(self, widget, event):
        """ Called automatically when the checkbox emits a StateChangedEvent (when the user clicks it) """
        state = bool(widget.GetRepresentation().GetState())
        if self.is_debug:
            print('OnUICheckboxChangedCallback(widget[{}]): updated value {})'.format(self.widget_index, state))
        widget.ProcessEventsOff()
        self.callback(self.widget_index, state) # perform the callback that will update the mutually exclusive active element using the widget changed callback
        widget.ProcessEventsOn()
      

class CallbackSequence:
    """ Helper class to call a list of callbacks with the same argument sequentally """
    def __init__(self, callbacks_list, is_debug=False):
        self.is_debug=is_debug
        self.callbacks_list = callbacks_list

    def __call__(self, state):    
        # call the callbacks in order
        for i in np.arange(len(self.callbacks_list)):
            self.callbacks_list[i](state)
      
        
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
    
        


# def add_batch_toggle_checkboxes(p, mesh_actors, colors, widget_check_states=False, widget_size=20, widget_start_pos=12, widget_border_size=3, additional_callback_actions=None, labels=None, base_widget_name='lblPlacemapCheckboxLabel'):
#     # """ Adds a list of toggle checkboxes to turn on and off each placemap"""
#     """ Adds a list of toggle checkboxes to turn on and off each placemap
#     Usage:
#         checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks = add_placemap_toggle_checkboxes(pActiveTuningCurvesPlotter, tuningCurvePlotActors, pf_colors, widget_check_states=True, is_debug=False)
#     """
#     num_checkboxes = len(mesh_actors)
#     if type(widget_check_states)==bool:
#         widget_check_states = np.full([num_checkboxes,], widget_check_states)

#     start_positions = widget_start_pos + ((widget_size + (widget_size // 10)) * np.flip(np.arange(num_checkboxes)))
#     checkboxWidgetActors = list()
#     labelWidgetActors = list()
    
#     # callbacks
#     # visibility_callbacks = list()
#     # checkboxWidget_IsChecked_callbacks = list()
#     combined_callbacks = list()
    
#     for i, an_actor in enumerate(mesh_actors):
#         curr_widget_position = (5.0, start_positions[i])
#         # Make a separate callback for each widget
#         curr_visibility_callback = SetVisibilityCallback(an_actor)
#         if additional_callback_actions is not None:
#             curr_custom_callback = additional_callback_actions[i]
#         else:
#             curr_custom_callback = None
            
#         # curr_visibility_callback(widget_check_states[i]) # perform the callback to update the initial visibility based on the correct state for this object
#         if labels is None:
#             curr_widget_label = '{}'.format(i)
#         else:
#             curr_widget_label = labels[i]
#         curr_widget_actor = PhoWidgetHelper.perform_add_custom_button_widget(p, curr_visibility_callback, value=widget_check_states[i],
#                 position=curr_widget_position, size=widget_size,
#                 border_size=widget_border_size,
#                 color_on=colors[:,i],
#                 color_off='grey',
#                 background_color=colors[:,i] # background_color is used for the border
#         )
#         # builds a widget label with a name like "lblPlacemapCheckboxLabel[5]"
#         curr_widget_label_actor = PhoWidgetHelper.perform_add_button_text_label(p, curr_widget_label, curr_widget_position, font_size=6, color=[1, 1, 1], shadow=False, name='{}[{}]'.format(base_widget_name, i), viewport=False)        
        
#         curr_checkbox_checked_callback = SetUICheckboxValueCallback(curr_widget_actor)
#         curr_combined_callback = CallbackSequence([curr_visibility_callback, curr_checkbox_checked_callback])
#         if curr_custom_callback is not None:
#             curr_combined_callback.callbacks_list.append(curr_custom_callback)
        
#         combined_callbacks.append(curr_combined_callback)
#         # append actors to lists:
#         labelWidgetActors.append(curr_widget_label_actor)
#         checkboxWidgetActors.append(curr_widget_actor)

#     return checkboxWidgetActors, combined_callbacks





def add_placemap_toggle_checkboxes(p, active_idx_updated_callbacks, colors, widget_check_states=False, widget_size=20, widget_start_pos=12, widget_border_size=3, additional_callback_actions=None, labels=None):
    # """ Adds a list of toggle checkboxes to turn on and off each placemap"""
    """ Adds a list of toggle checkboxes to turn on and off each placemap
    Usage:
        checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks = add_placemap_toggle_checkboxes(pActiveTuningCurvesPlotter, tuningCurvePlotActors, pf_colors, widget_check_states=True, is_debug=False)
    """
    num_checkboxes = len(active_idx_updated_callbacks)
    if type(widget_check_states)==bool:
        widget_check_states = np.full([num_checkboxes,], widget_check_states)

    start_positions = widget_start_pos + ((widget_size + (widget_size // 10)) * np.flip(np.arange(num_checkboxes)))
    checkboxWidgetActors = list()
    labelWidgetActors = list()
    
    # callbacks
    final_combined_callbacks = list()
    
    for i, curr_callback in enumerate(active_idx_updated_callbacks):
        curr_widget_position = (5.0, start_positions[i])
        # Make a separate callback for each widget
        if additional_callback_actions is not None:
            curr_custom_callback = additional_callback_actions[i]
        else:
            curr_custom_callback = None
            
        curr_callback(widget_check_states[i]) # perform the callback to update the initial visibility based on the correct state for this object
        
        curr_widget_actor = PhoWidgetHelper.perform_add_custom_button_widget(p, curr_callback, value=widget_check_states[i],
                position=curr_widget_position, size=widget_size,
                border_size=widget_border_size,
                color_on=colors[:,i],
                color_off='grey',
                background_color=colors[:,i], # background_color is used for the border
                render=True
        )
        curr_widget_label_actor = PhoWidgetHelper.perform_add_button_text_label(p, '{}'.format(i), curr_widget_position, font_size=6, color=[1, 1, 1], shadow=False, name='lblPlacemapCheckboxLabel[{}]'.format(i), viewport=False)        
        curr_checkbox_checked_callback = SetUICheckboxValueCallback(curr_widget_actor)
        if isinstance(curr_callback, CallbackSequence):
            curr_combined_callback = curr_callback
            curr_combined_callback.callbacks_list.append(curr_checkbox_checked_callback)
        else:
            curr_combined_callback = CallbackSequence([curr_callback, curr_checkbox_checked_callback]) # otherwise if it isn't a CallbackSequence, we just have to wrap it in one
            
        if curr_custom_callback is not None:
            curr_combined_callback.callbacks_list.append(curr_custom_callback) 
        # append the callbacks to the lists:
        
        final_combined_callbacks.append(curr_combined_callback)
        # append actors to lists:
        labelWidgetActors.append(curr_widget_label_actor)
        checkboxWidgetActors.append(curr_widget_actor)

    return checkboxWidgetActors, final_combined_callbacks



def add_placemap_toggle_mutually_exclusive_checkboxes(p, active_idx_updated_callbacks, colors, active_element_idx=0, widget_size=20, widget_start_pos=12, widget_border_size=3, require_active_selection=False, is_debug=False, debug_console_widget=None, additional_callback_actions=None, labels=None):
    """ Adds a list of toggle checkboxes that only allows one at a time to be selected to turn on and off each placemap
    Usage:
        checkboxWidgetActors, tuningCurvePlotActorVisibilityCallbacks, mutually_exclusive_radiobutton_group = add_placemap_toggle_mutually_exclusive_checkboxes(pActiveTuningCurvesPlotter, tuningCurvePlotActors, pf_colors, active_element_idx=4, require_active_selection=False, is_debug=False)
    """
    num_checkboxes = len(active_idx_updated_callbacks)
    # start_positions = widget_start_pos + ((widget_size + (widget_size // 10)) * np.arange(num_checkboxes))
    start_positions = widget_start_pos + ((widget_size + (widget_size // 10)) * np.flip(np.arange(num_checkboxes)))
    
    checkboxWidgetActors = list()
    labelWidgetActors = list()
    
    # callbacks
    final_combined_callbacks = list()
    
    for i, curr_callback in enumerate(active_idx_updated_callbacks):
        curr_widget_position = (5.0, start_positions[i])
        if additional_callback_actions is not None:
            curr_custom_callback = additional_callback_actions[i]
        else:
            curr_custom_callback = None
            
        curr_widget_actor = PhoWidgetHelper.perform_add_custom_button_widget(p, curr_callback, value=False,
                position=curr_widget_position, size=widget_size,
                border_size=widget_border_size,
                color_on=colors[:,i],
                color_off='grey',
                background_color=colors[:,i], # background_color is used for the border
                render=True
        )
        
        if labels is None:
            curr_widget_label = '{}'.format(i)
        else:
            curr_widget_label = labels[i]
            
        curr_widget_label_actor = PhoWidgetHelper.perform_add_button_text_label(p, curr_widget_label, curr_widget_position, font_size=6, color=[1, 1, 1], shadow=False, name='lblPlacemapCheckboxLabel[{}]'.format(i), viewport=False)        
        curr_checkbox_checked_callback = SetUICheckboxValueCallback(curr_widget_actor)
        if isinstance(curr_callback, CallbackSequence):
            curr_combined_callback = curr_callback
            curr_combined_callback.callbacks_list.append(curr_checkbox_checked_callback)
        else:
            curr_combined_callback = CallbackSequence([curr_callback, curr_checkbox_checked_callback]) # otherwise if it isn't a CallbackSequence, we just have to wrap it in one
            
        if curr_custom_callback is not None:
            curr_combined_callback.callbacks_list.append(curr_custom_callback) 
        # append the callbacks to the lists:
        
        final_combined_callbacks.append(curr_combined_callback)
        # append actors to lists:
        labelWidgetActors.append(curr_widget_label_actor)
        checkboxWidgetActors.append(curr_widget_actor)

    
    # build the mutually exclusive group:
    mutually_exclusive_radiobutton_group = MutuallyExclusiveRadioButtonGroup(len(final_combined_callbacks), active_element_idx=active_element_idx, on_element_state_changed_callbacks=final_combined_callbacks, require_active_selection=require_active_selection, is_debug=is_debug)
    
    def _update_mutually_exclusive_callback(widget_index, state):
        if debug_console_widget is not None:
            debug_console_widget.add_line_to_buffer('_update_mutually_exclusive_callback(widget[{}]): updated value {})'.format(widget_index, state))
        mutually_exclusive_radiobutton_group[widget_index] = state # set the mutually exclusive active element using the widget changed callback

    # add the function that responds to user initiated changes by clicking on the value
    # checkbox_changed_callbacks = list()
    for i, a_checkbox_widget_actor in enumerate(checkboxWidgetActors):
        curr_checkbox_changed_callback = OnUICheckboxChangedCallback(a_checkbox_widget_actor, i, _update_mutually_exclusive_callback, is_debug=is_debug)
        a_checkbox_widget_actor.AddObserver(pv._vtk.vtkCommand.StateChangedEvent, curr_checkbox_changed_callback)
        # checkbox_changed_callbacks.append(curr_checkbox_changed_callback)
        
    return checkboxWidgetActors, final_combined_callbacks, mutually_exclusive_radiobutton_group