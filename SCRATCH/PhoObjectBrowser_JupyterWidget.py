from attrs import define, field
from ipywidgets import Accordion, Label
from IPython.display import display

@define(slots=False)
class ObjectBrowser:
    """ 
    from Spike3D.SCRATCH.PhoObjectBrowser_JupyterWidget import ObjectBrowser
    
    """
    object_to_browse: object

    def display_object(self, obj, level=0):
        accordion = Accordion(children=[])
        accordion.selected_index = None # Start with all collapsed

        for key, value in vars(obj).items():
            label = Label(f'{key}: {value}')
            accordion.children += (label,)
            accordion.set_title(level, key)
            
            if hasattr(value, '__dict__'):
                inner_accordion = self.display_object(value, level+1)
                accordion.children += (inner_accordion,)

        return accordion

    def show(self):
        root_accordion = self.display_object(self.object_to_browse)
        display(root_accordion)
        