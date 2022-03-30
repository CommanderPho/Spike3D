

class AnimationStateBaseMixin:
    """ used by AnimationStateVTKMixin (specialized for VTK) which is in-turn used by PhoInteractivePlotter 
    
    Required Properties:
        self.interface_properties.animation_state
    """
    # define the animation switch
    def toggle_animation(self, state):
        self.interface_properties.animation_state = state # updates the animation state to the new value
    
    def add_ui(self):
        """ VTK A checkbox that decides whether we're playing back at a constant rate or not."""
        raise NotImplementedError
