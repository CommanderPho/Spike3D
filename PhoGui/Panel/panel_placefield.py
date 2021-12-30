import param
import panel as pn
from panel.viewable import Viewer

from PhoPositionalData.plotting.mixins.placefield_plotting_mixins import ActivePlacefieldsPlotting, SinglePlacefieldPlottingExtended



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


class SingleEditablePlacefieldDisplayConfiguration(SinglePlacefieldPlottingExtended, Viewer):
    """ Panel configuration for a single placefield display (as in for a single cell)
    Usage:
        single_editable_pf_custom_widget = SingleEditablePlacefieldDisplayConfiguration(ipcDataExplorer.active_tuning_curve_render_configs[2])
        single_editable_pf_custom_widget
    """
    # config = SinglePlacefieldPlottingExtended()
    
    # value = param.Range(doc="A numeric range.")
    # width = param.Integer(default=300)
    
    def __init__(self, config=None, callbacks=None, **params):
        if config is not None:
            self.name = config.name
            self.color = config.color
            self.isVisible = config.isVisible
            self.spikesVisible = config.spikesVisible
            
        if callbacks is not None:
            assert isinstance(callbacks, dict), "callbacks argument should be a dictionary with keys 'pf' and 'spikes'!"
            self._callbacks = callbacks
        else:
            self._callbacks = None
            raise ValueError
        
        # self._start_input = pn.widgets.FloatInput()
        # self._end_input = pn.widgets.FloatInput(align='end')
        self._wgt_label_button = pn.widgets.Button(name=self.name, button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
        self._wgt_color_picker = pn.widgets.ColorPicker(value=self.color, width=60, height=20, margin=0)
        self._wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=self.isVisible, margin=0)
        self._wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=self.spikesVisible, margin=0)
        super().__init__(**params)
        # Output Grid:
        self._layout = pn.GridSpec(width=60, height=100, margin=0)
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
        
    @param.depends('_wgt_label_button.name', '_wgt_color_picker.value', '_wgt_toggle_visible.value', '_wgt_toggle_spikes.value', watch=True)
    def _sync_params(self):
        self.name = self._wgt_label_button.name
        self.color = self._wgt_color_picker.value
        self.isVisible = self._wgt_toggle_visible.value
        self.spikesVisible = self._wgt_toggle_spikes.value
        
    @param.depends('_wgt_toggle_visible.value', watch=True)
    def _on_toggle_plot_visible_changed(self):
        print('_on_toggle_plot_visible_changed(...)')
        if self._callbacks is not None:
            self._callbacks['pf'](self.config_from_state()) # get the config from the updated state
            # self._callbacks(self.config_from_state()) # get the config from the updated state
        else:
            print('WARNING: no callback defined for pf value changes!')
            
    
    @param.depends('_wgt_toggle_spikes.value', watch=True)
    def _on_toggle_spikes_visible_changed(self):
        print('_on_toggle_spikes_visible_changed(...)')
        if self._callbacks is not None:
            updated_config = self.spikesVisible
            self._callbacks['spikes'](bool(self.spikesVisible)) # get the config from the updated state
            # self._callbacks(self.config_from_state()) # get the config from the updated state
        else:
            print('WARNING: no callback defined for spikes value changes!')
            
            
    def update_from_config(self, config):
        self.name = config.name
        self.color = config.color
        self.isVisible = config.isVisible
        self.spikesVisible = config.spikesVisible

    
    def config_from_state(self):
        return SinglePlacefieldPlottingExtended(name=self.name, isVisible=self.isVisible, color=self.color, spikesVisible=self.spikesVisible)

    @classmethod
    def build_all_placefield_output_panels(cls, configs, tuning_curve_config_changed_callback, spikes_config_changed_callback):
        """[summary]

        Args:
            configs ([type]): as would be obtained from ipcDataExplorer.active_tuning_curve_render_configs

        Returns:
            [type]: [description]
            
        Usage:
            out_panels = SingleEditablePlacefieldDisplayConfiguration.build_all_placefield_output_panels(ipcDataExplorer.active_tuning_curve_render_configs)
            pn.Row(*out_panels, height=120)        
        """
        
        # @param.depends(c.param.country, d.param.i, watch=True)
        # def g(country, i):
        #     print(f"g country={country} i={i}")

        out_panels = [SingleEditablePlacefieldDisplayConfiguration(config=a_config,
                                                                   callbacks={'pf':(lambda updated_config_copy=a_config, i_copy=idx: tuning_curve_config_changed_callback([i_copy], [updated_config_copy])),
                                                                              'spikes': (lambda are_included, i_copy=idx: spikes_config_changed_callback(cell_IDXs=[i_copy], cell_IDs=None, are_included=are_included))
                                                                              }) for (idx, a_config) in enumerate(configs)]
        return out_panels
        


def build_all_placefield_output_panels(ipcDataExplorer):
    out_panels = SingleEditablePlacefieldDisplayConfiguration.build_all_placefield_output_panels(ipcDataExplorer.active_tuning_curve_render_configs,
                                                                                                 tuning_curve_config_changed_callback=ipcDataExplorer.on_update_tuning_curve_display_config,
                                                                                                 spikes_config_changed_callback=ipcDataExplorer.change_unit_spikes_included)
    out_panels = pn.Row(*out_panels, height=120)
    return out_panels



class ActivePlacefieldsPlottingPanel(ActivePlacefieldsPlotting):
    """ Draws a selector for the active placefields to plot using two adjacent list controls.
    Usage:
        active_new_pf_panel = ActivePlacefieldsPlottingPanel(np.arange(ipcDataExplorer.num_tuning_curve_plot_actors), ipcDataExplorer.visible_tuning_curve_indicies)
        active_new_pf_panel.panel()
    """
    
    should_update_on_value_change = True
    
    def __init__(self, pf_option_indicies, pf_option_selected_values, num_pfs=None, update_included_cell_Indicies_callback=None, **params):
        # super(ActivePlacefieldsPlottingPanel, self).__init__(num_pfs=num_pfs, **params)
        super(ActivePlacefieldsPlottingPanel, self).__init__(**params)
        self.final_update_included_cell_Indicies_callback = None
        if update_included_cell_Indicies_callback is not None:
            if callable(update_included_cell_Indicies_callback):
                self.final_update_included_cell_Indicies_callback = update_included_cell_Indicies_callback
        
        assert (self.final_update_included_cell_Indicies_callback is not None), "An update_included_cell_Indicies_callback(x) callback is needed."
                
        if pf_option_indicies is not None:
            self.pf_option_indicies = pf_option_indicies
            self.num_pfs = len(pf_option_indicies)
        else:
            self.pf_option_indicies = np.arange(num_pfs)
            self.num_pfs = num_pfs
        if pf_option_selected_values is not None:
            self.pf_option_selected_values = pf_option_selected_values
        else:
            self.pf_option_selected_values = []

    def on_hide_all_placefields(self):
        print('on_hide_all_placefields()')
        self.pf_option_selected_values = []
        self.final_update_included_cell_Indicies_callback([])

    def on_update_active_placefields(self, updated_pf_indicies):
        print(f'on_update_active_placefields({updated_pf_indicies})')
        self.pf_option_selected_values = updated_pf_indicies
        self.final_update_included_cell_Indicies_callback(updated_pf_indicies)

    def btn_hide_all_callback(self, event):
        print('btn_hide_all_callback(...)')
        self.on_hide_all_placefields()

    def btn_update_active_placefields(self, event):
        print('btn_update_active_placefields(...)')
        updated_pf_options_list_ints = ActivePlacefieldsPlottingPanel.options_to_int(self.cross_selector.value) # convert to ints
        self.on_update_active_placefields(updated_pf_options_list_ints)

    def index_selection_changed_callback(self, *events):
        # print(events)
        for event in events:
            if event.name == 'options':
                self.selections.object = 'Possible options: %s' % ', '.join(event.new)
                self.pf_option_indicies = ActivePlacefieldsPlottingPanel.options_to_int(event.new) # convert to ints
                self.num_pfs = len(self.pf_option_indicies)

            elif event.name == 'value':
                if ActivePlacefieldsPlottingPanel.should_update_on_value_change:
                    updated_pf_options_list_ints = ActivePlacefieldsPlottingPanel.options_to_int(event.new) # convert to ints
                    self.on_update_active_placefields(updated_pf_options_list_ints)
                self.selected.object = 'Selected: %s' % ','.join(event.new)

    def panel(self):
        # Panel pane and widget objects:
        self.selections = pn.pane.Markdown(object='')
        self.selected = pn.pane.Markdown(object='')
        self.cross_selector = pn.widgets.CrossSelector(name='Active Placefields', value=[], options=['0', '1', '2'], height=600, width=200) # cross_selector.value

        # Action Buttons:
        self.button_hide_all = pn.widgets.Button(name='Hide All Placefields')
        self.button_hide_all.on_click(self.btn_hide_all_callback)
        self.button_update = pn.widgets.Button(name='Update Active Placefields', button_type='primary')
        self.button_update.on_click(self.btn_update_active_placefields)

        self.watcher = self.cross_selector.param.watch(self.index_selection_changed_callback, ['options', 'value'], onlychanged=False)
        # set initial
        # active_new_pf_panel.set_initial(self.num_pfs, [0, 1, 5])
        self.set_initial(self.pf_option_indicies, self.pf_option_selected_values, num_pfs=self.num_pfs)

        return pn.Column(pn.Row(self.cross_selector, width=200, height=600),
                         pn.Spacer(width=200, height=10),
                         self.selections,
                         pn.Spacer(width=200, height=10),
                         self.selected,
                         pn.Spacer(width=200, height=20),
                         pn.Row(self.button_hide_all, self.button_update)
                        )

    def set_initial(self, option_values, selected_values, num_pfs=None):
        # set initial
        if option_values is None:
            pf_options_list_ints, pf_options_list_strings = ActivePlacefieldsPlotting.build_pf_options_list(num_pfs)
        else:
            pf_options_list_strings = ActivePlacefieldsPlottingPanel.options_to_str(option_values)
        options = pf_options_list_strings
        # selected_values = [str(an_item) for an_item in selected_values]
        selected_values = ActivePlacefieldsPlottingPanel.options_to_str(selected_values)
        self.cross_selector.param.set_param(options=dict(zip(options, options)), value=selected_values)




# class SingleEditablePlacefieldDisplayConfiguration(SinglePlacefieldPlottingExtended, Viewer):
#     # config = SinglePlacefieldPlottingExtended()
    
#     # value = param.Range(doc="A numeric range.")
#     # width = param.Integer(default=300)
    
#     def __init__(self, **params):
#         # self._start_input = pn.widgets.FloatInput()
#         # self._end_input = pn.widgets.FloatInput(align='end')
#         self._wgt_label_button = pn.widgets.Button(name=f'pf[{self.config.name}]', button_type='default', margin=0, height=20, sizing_mode='stretch_both', width_policy='min')
#         self._wgt_color_picker = pn.widgets.ColorPicker(value=self.config.color, width=60, height=20, margin=0)
#         self._wgt_toggle_visible = pn.widgets.Toggle(name='isVisible', value=self.config.isVisible, margin=0)
#         self._wgt_toggle_spikes = pn.widgets.Toggle(name='SpikesVisible', value=self.config.spikesVisible, margin=0)
#         super().__init__(**params)
#         # Output Grid:
#         self._layout = pn.GridSpec(width=100, height=100, margin=0)
#         self._layout[0, :3] = self._wgt_label_button
#         self._layout[1, :] = self._wgt_color_picker
#         self._layout[2, :] = pn.Row(self._wgt_toggle_visible, margin=0, background='red')
#         self._layout[3, :] = pn.Row(self._wgt_toggle_spikes, margin=0, background='green')

#         # self._layout = pn.Row(self._start_input, self._end_input)
#         self._sync_widgets()
    
#     def __panel__(self):
#         return self._layout
    
#     # @param.depends('config', watch=True)
#     # @param.depends('config.name','config.color','config.isVisible','config.spikesVisible', watch=True)
#     @param.depends('config', watch=True)
#     def _sync_widgets(self):
#         self._wgt_label_button.name = f'pf[{self.config.name}]'
#         self._wgt_color_picker.value = self.config.color
#         self._wgt_toggle_visible.value = self.config.isVisible
#         self._wgt_toggle_spikes.value = self.config.spikesVisible
        
#     @param.depends('_wgt_color_picker.value', '_wgt_toggle_visible.value', '_wgt_toggle_spikes.value', watch=True)
#     def _sync_params(self):
#         self.config.color = self._wgt_color_picker.value
#         self.config.isVisible = self._wgt_toggle_visible.value
#         self.config.spikesVisible = self._wgt_toggle_spikes.value



# single_editable_pf_custom_widget = SingleEditablePlacefieldDisplayConfiguration(config=ipcDataExplorer.active_tuning_curve_render_configs[0])
# single_editable_pf_custom_widget
# pn.Row(SingleEditablePlacefieldDisplayConfiguration.param)