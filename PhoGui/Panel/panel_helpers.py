import param
import panel as pn
from panel.viewable import Viewer
from PhoPositionalData.plotting.mixins.general_plotting_mixins import ActivePlacefieldsPlotting, SinglePlacefieldPlottingExtended



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





def build_single_placefield_output_panel(render_config):
    """ An alternative to the whole SingleEditablePlacefieldDisplayConfiguration implementation """
    wgt_label_button = pn.widgets.Button(name=f'pf[{render_config.name}]', button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
    wgt_color_picker = pn.widgets.ColorPicker(value=render_config.color, width=60, height=20, margin=0)
    wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=render_config.isVisible, margin=0)
    wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=render_config.spikesVisible, margin=0)

    # gspec = pn.GridSpec(sizing_mode='stretch_both', max_height=800)
    # Output Grid:
    gspec = pn.GridSpec(width=100, height=100, margin=0)
    gspec[0, :3] = wgt_label_button
    gspec[1, :] = wgt_color_picker
    gspec[2, :] = pn.Row(wgt_toggle_visible, margin=0, background='red')
    gspec[3, :] = pn.Row(wgt_toggle_spikes, margin=0, background='green')
    return gspec


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




class SingleEditablePlacefieldDisplayConfiguration(SinglePlacefieldPlottingExtended, Viewer):
    """ 
    
    Usage:
		single_editable_pf_custom_widget = SingleEditablePlacefieldDisplayConfiguration(ipcDataExplorer.active_tuning_curve_render_configs[2])
		single_editable_pf_custom_widget
    """
    # config = SinglePlacefieldPlottingExtended()
    
    # value = param.Range(doc="A numeric range.")
    # width = param.Integer(default=300)
    
    def __init__(self, config=None, **params):
        if config is not None:
            self.name = config.name
            self.color = config.color
            self.isVisible = config.isVisible
            self.spikesVisible = config.spikesVisible
        
        # self._start_input = pn.widgets.FloatInput()
        # self._end_input = pn.widgets.FloatInput(align='end')
        self._wgt_label_button = pn.widgets.Button(name=self.name, button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
        self._wgt_color_picker = pn.widgets.ColorPicker(value=self.color, width=60, height=20, margin=0)
        self._wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=self.isVisible, margin=0)
        self._wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=self.spikesVisible, margin=0)
        super().__init__(**params)
        # Output Grid:
        self._layout = pn.GridSpec(width=100, height=100, margin=0)
        self._layout[0, :3] = self._wgt_label_button
        self._layout[1, :] = self._wgt_color_picker
        self._layout[2, :] = pn.Row(self._wgt_toggle_visible, margin=0, background='red')
        self._layout[3, :] = pn.Row(self._wgt_toggle_spikes, margin=0, background='green')
        if config is not None:
            self.update_from_config(config)
        
        self._sync_widgets()
    
    def __panel__(self):
        return self._layout
    
    # @param.depends('config', watch=True)
    # @param.depends('config.name','config.color','config.isVisible','config.spikesVisible', watch=True)
    # @param.depends('config', watch=True)
    @param.depends('name','color','isVisible','spikesVisible', watch=True)
    def _sync_widgets(self):
        self._wgt_label_button.name = self.name
        self._wgt_color_picker.value = self.color
        self._wgt_toggle_visible.value = self.isVisible
        self._wgt_toggle_spikes.value = self.spikesVisible
        
    @param.depends('_wgt_color_picker.value', '_wgt_toggle_visible.value', '_wgt_toggle_spikes.value', watch=True)
    def _sync_params(self):
        self.color = self._wgt_color_picker.value
        self.isVisible = self._wgt_toggle_visible.value
        self.spikesVisible = self._wgt_toggle_spikes.value

    def update_from_config(self, config):
        self.name = config.name
        self.color = config.color
        self.isVisible = config.isVisible
        self.spikesVisible = config.spikesVisible

    def config_from_state(self):
        return SinglePlacefieldPlottingExtended(name=self.name, isVisible=self.isVisible, color=self.color, spikesVisible=self.spikesVisible)

        


def build_all_placefield_output_panels(ipcDataExplorer):
    # out_panels = [build_single_placefield_output_panel(a_config) for a_config in ipcDataExplorer.active_tuning_curve_render_configs]
    out_panels = [SingleEditablePlacefieldDisplayConfiguration(config=a_config) for a_config in ipcDataExplorer.active_tuning_curve_render_configs]
    return out_panels
