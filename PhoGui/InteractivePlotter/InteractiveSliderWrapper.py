#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho

"""

## TODO: very VTK centric, but doesn't need to be

class InteractiveSliderWrapper:
    """ a wrapper around a VTK GUI slider widget that can be used to sync state between the slider and the pyvista plotter that it controls. """
    # instance attributes
    def __init__(self, slider_obj):
        self.slider_obj = slider_obj
        # curr_min_value = self.slider_obj.GetRepresentation().GetMinimumValue()
        # curr_max_value = self.slider_obj.GetRepresentation().GetMaximumValue()
        # curr_value = self.slider_obj.GetRepresentation().GetValue()

    @property
    def curr_value(self):
        """ VTK: The curr_value property."""
        return self.slider_obj.GetRepresentation().GetValue()
    
    @curr_value.setter
    def curr_value(self, value):
        self.update_value(value)


    @property
    def curr_index(self):
        """The curr_index property."""
        return int(self.curr_value)
    
    @curr_index.setter
    def curr_index(self, value):
        self.update_value(float(value))


    # instance method
    def update_value(self, new_value):
        """ VTK """
        from pyvista import _vtk
        self.slider_obj.GetRepresentation().SetValue(new_value)
        self.slider_obj.InvokeEvent(_vtk.vtkCommand.InteractionEvent) # Called to ensure user callback is performed.


    def _safe_constrain_index(self, proposed_index):
        """ VTK """
        curr_min_value = self.slider_obj.GetRepresentation().GetMinimumValue()
        curr_max_value = self.slider_obj.GetRepresentation().GetMaximumValue()
        
        if (proposed_index < curr_min_value):
            print('value too low!')
            proposed_index = curr_min_value
        elif (curr_max_value < proposed_index):
            print('value too high!')
            proposed_index = curr_max_value
        else:
            # proposed_index is within the range and is fine
            pass
        return proposed_index
    
        
    def step_index(self, step_size):
        curr_index = self.curr_index
        proposed_index = self._safe_constrain_index(curr_index + step_size)
        self.curr_index = proposed_index
        return proposed_index
    
    def step_next_index(self):
        return self.step_index(1)
        
    def step_prev_index(self):
        return self.step_index(-1) 

