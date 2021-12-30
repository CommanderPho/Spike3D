import param
import panel as pn
from panel.viewable import Viewer


def build_carousel_scroller(items):
    """ 
    Usage:
        item_layout = widgets.Layout(height='120px', min_width='40px')
        items = [pn.Row(a_widget, layout=item_layout, margin=0, background='black') for a_widget in single_pf_output_panels]
        # items = [widgets.Button(layout=item_layout, description=str(i), button_style='success') for i in range(40)]
        # build_carousel_scroller(items)
        build_carousel_scroller(single_pf_output_panels)
    """
    box_layout = pn.widgets.Layout(overflow_x='scroll', border='3px solid black',
                        width='1024px',
                        height='',
                        flex_flow='row',
                        display='flex')
    carousel = pn.widgets.Box(children=items, layout=box_layout)
    return pn.widgets.VBox([pn.widgets.Label('Scroll horizontally:'), carousel])






# class EditableRange(Viewer):    
#	""" An example Viewer implementation """
#     value = param.Range(doc="A numeric range.")
#     width = param.Integer(default=300)
    
#     def __init__(self, **params):
#         self._start_input = pn.widgets.FloatInput()
#         self._end_input = pn.widgets.FloatInput(align='end')
#         super().__init__(**params)
#         self._layout = pn.Row(self._start_input, self._end_input)
#         self._sync_widgets()
    
#     def __panel__(self):
#         return self._layout
    
#     @param.depends('value', 'width', watch=True)
#     def _sync_widgets(self):
#         self._start_input.name = self.name
#         self._start_input.value = self.value[0]
#         self._end_input.value = self.value[1]
#         self._start_input.width = self.width//2
#         self._end_input.width = self.width//2
        
#     @param.depends('_start_input.value', '_end_input.value', watch=True)
#     def _sync_params(self):
#         self.value = (self._start_input.value, self._end_input.value)

# range_widget = EditableRange(name='Range', value=(0, 10))



