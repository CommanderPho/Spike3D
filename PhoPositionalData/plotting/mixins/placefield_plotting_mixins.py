import param
import numpy as np
import pandas as pd

from PhoPositionalData.plotting.mixins.general_plotting_mixins import NeuronConfigOwningMixin, OptionsListMixin
from neuropy.core.neuron_identities import NeuronIdentityAccessingMixin


class PlacefieldOwningMixin(NeuronIdentityAccessingMixin, NeuronConfigOwningMixin):
    """ Implementor owns placefields and has access to their data and configuration objects """
    debug_logging = False
    
    @property
    def placefields(self):
        return self.params.active_epoch_placefields
    
    @property
    def ratemap(self):
        return self.placefields.ratemap
    
    @property
    def tuning_curves(self):
        return self.ratemap.normalized_tuning_curves
    
    @property
    def num_tuning_curves(self):
        return len(self.tuning_curves)
    
    @property
    def tuning_curve_indicies(self):
        return np.arange(self.num_tuning_curves)
    
    @property
    def active_tuning_curve_render_configs(self):
        """The active_tuning_curve_render_configs property."""
        return self.active_neuron_render_configs
    @active_tuning_curve_render_configs.setter
    def active_tuning_curve_render_configs(self, value):
        self.active_neuron_render_configs = value
        
    def build_tuning_curve_configs(self):
        # call the parent function
        self.build_neuron_render_configs()
        # do any addition setup needed
            
    # @property
    # def active_tuning_curve_render_configs(self):
    #     """The active_tuning_curve_render_configs property."""
    #     return self.params.pf_active_configs
    # @active_tuning_curve_render_configs.setter
    # def active_tuning_curve_render_configs(self, value):
    #     self.params.pf_active_configs = value
        
    # def build_tuning_curve_configs(self):
    #     # Get the cell IDs that have a good place field mapping:
    #     good_placefield_neuronIDs = np.array(self.ratemap.neuron_ids) # in order of ascending ID
    #     unit_labels = [f'{good_placefield_neuronIDs[i]}' for i in np.arange(self.num_tuning_curves)]
    #     self.active_tuning_curve_render_configs = [SingleNeuronPlottingExtended(name=unit_labels[i], isVisible=False, color=self.params.pf_colors_hex[i], spikesVisible=False) for i in self.tuning_curve_indicies]



    
class HideShowPlacefieldsRenderingMixin(PlacefieldOwningMixin):
    """ Implementor Visually Displays Placefield data and enables basic interactivity for it. """
    debug_logging = False
        
    @property
    def tuning_curve_plot_actors(self):
        return self.plots['tuningCurvePlotActors']
    
    @property
    def num_tuning_curve_plot_actors(self):
        return len(self.tuning_curve_plot_actors)
    
    @property
    def tuning_curve_is_visible(self):
        return np.array([bool(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors], dtype=bool)
        
    @property
    def tuning_curve_visibilities(self):
        return np.array([int(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors], dtype=int)

    @property
    def visible_tuning_curve_indicies(self):
        all_indicies = np.arange(self.num_tuning_curve_plot_actors)
        return all_indicies[self.tuning_curve_is_visible]
    
    def update_active_placefields(self, placefield_indicies):
        """ 
        Usage: 
            included_cell_ids = [48, 61]
            included_cell_INDEXES = [ipcDataExplorer.get_neuron_id_and_idx(cell_id=an_included_cell_ID)[0] for an_included_cell_ID in included_cell_ids] # get the indexes from the cellIDs
            ipcDataExplorer.update_active_placefields(included_cell_INDEXES) # actives only the placefields that have aclu values (cell ids) in the included_cell_ids array.
        """
        self._hide_all_tuning_curves() # hide all tuning curves to begin with (for a fresh slate)
        for a_pf_idx in placefield_indicies:
            self._show_tuning_curve(a_pf_idx)
        
    def _hide_all_tuning_curves(self):
        # Works to hide all turning curve plots:
        for aTuningCurveActor in self.tuning_curve_plot_actors:
            aTuningCurveActor.SetVisibility(0)

    def _show_all_tuning_curves(self):
        # Works to show all turning curve plots by updating the render configs:
        print('WARNING: _show_all_tuning_curves() does not currently work.')
        tuning_curve_config_indicies = np.arange(self.num_tuning_curves)
        # update the configs:
        curr_configs = self.active_tuning_curve_render_configs       
        for config_idx in tuning_curve_config_indicies:
            curr_configs[config_idx].isVisible = True
        # print(f'curr_configs: {curr_configs}')
        self.on_update_tuning_curve_display_config(tuning_curve_config_indicies, curr_configs)

            
    def _show_tuning_curve(self, show_index):
        # Works to show the specified tuning curve plots:
        self.tuning_curve_plot_actors[show_index].SetVisibility(1)
        
    def on_update_tuning_curve_display_config(self, updated_config_indicies, updated_configs):
        if self.debug_logging:
            print(f'HideShowPlacefieldsRenderingMixin.on_update_tuning_curve_display_config(updated_config_indicies: {updated_config_indicies}, updated_configs: {updated_configs})')
        assert hasattr(self, 'update_neuron_render_configs'), "self must be of type NeuronConfigOwningMixin to have access to its configs"
        self.update_neuron_render_configs(updated_config_indicies, updated_configs) # update the config with the new values:
        for an_updated_config_idx, an_updated_config in zip(updated_config_indicies, updated_configs):
            self.tuning_curve_plot_actors[an_updated_config_idx].SetVisibility(int(self.active_tuning_curve_render_configs[an_updated_config_idx].isVisible)) # update visibility of actor
            
        
    def update_tuning_curve_configs(self):
        for i, aTuningCurveActor in enumerate(self.tuning_curve_plot_actors):
            self.active_tuning_curve_render_configs[i].isVisible = bool(aTuningCurveActor.GetVisibility())
            
    def apply_tuning_curve_configs(self):
        for i, aTuningCurveActor in enumerate(self.tuning_curve_plot_actors):
            aTuningCurveActor.SetVisibility(int(self.active_tuning_curve_render_configs[i].isVisible))




        
        
class ActivePlacefieldsPlotting(OptionsListMixin, param.Parameterized):
    """ """
    # _on_hide_all_placefields = lambda x: print(f'_on_hide_all_placefields({x})')
    # _on_update_active_placefields = lambda x: print(f'_on_update_active_placefields({x})')
    
    # pf_options_list_ints, pf_options_list_strings = ActivePlacefieldsPlotting.build_pf_options_list(40)

    # curr_active_pf_idx_list_label = param.Parameter(default="[]", constant=True)
    curr_active_pf_idx_list_label = param.Parameter(default="[]", constant=False)
    active_pf_idx_list = param.ListSelector(default=[], objects= ['0', '1', '2'], precedence=0.5)
    dictionary = param.Dict(default={"a": 2, "b": 9})
    
    # hide_all_action = param.Action(lambda selected_indicies: self.on_hide_all_placefields(), doc="""Hide All Placefields.""", precedence=0.7)
    # update_action = param.Action(lambda selected_indicies: self.on_update_active_placefields([int(an_idx) for an_idx in selected_indicies]), doc="""Update Placefield Visibility.""", precedence=0.7)
    # active_pf_idx_list = pn.widgets.CrossSelector(name='Active Placefields', value=[], options=pf_options_list_strings) # cross_selector.value
    
    
    def __init__(self, **params):
        super(ActivePlacefieldsPlotting, self).__init__(**params)
       
    def on_hide_all_placefields(self):
        print('on_hide_all_placefields()')
        # lambda x: print(f'_on_hide_all_placefields({x})')

    def on_update_active_placefields(self, updated_pf_indicies):
        print(f'on_update_active_placefields({updated_pf_indicies})')
 
    # def index_selection_changed_callback(self, *events):
    #     print(events)
    #     for event in events:
    #         if event.name == 'options':
    #             self.selections.object = 'Possible options: %s' % ', '.join(event.new)
    #         elif event.name == 'value':
    #             self.selected.object = 'Selected: %s' % ','.join(event.new)
  
    @param.depends('active_pf_idx_list', watch=True)
    def _update_curr_active_pf_idx_list_label(self):        
        flat_updated_idx_list_string = ','.join(self.active_pf_idx_list)
        flat_updated_idx_list_string = f'[{flat_updated_idx_list_string}]'
        print('flat_updated_idx_list_string: {flat_updated_idx_list_string}')
        self.curr_active_pf_idx_list_label = flat_updated_idx_list_string


# class ActivePlacefieldsPlottingNew:
    
#     def __init__(self, num_pfs, **params):
#         super(ActivePlacefieldsPlottingNew, self).__init__(**params)
#         self.num_pfs = num_pfs
#         # self.figure = figure(x_range=(-1, 1), y_range=(-1, 1))
#         # self.renderer = self.figure.line(*self._get_coords())
    
#     @staticmethod
#     def build_pf_options_list(num_pfs=40):
#         pf_options_list_ints = np.arange(num_pfs)
#         pf_options_list_strings = [f'{i}' for i in pf_options_list_ints]
#         return pf_options_list_ints, pf_options_list_strings
    
#     # _on_hide_all_placefields = lambda x: print(f'_on_hide_all_placefields({x})')
#     # _on_update_active_placefields = lambda x: print(f'_on_update_active_placefields({x})')
    
#     def on_hide_all_placefields(self):
#         print('on_hide_all_placefields()')
#         # lambda x: print(f'_on_hide_all_placefields({x})')
    
#     def on_update_active_placefields(self, updated_pf_indicies):
#         print(f'on_update_active_placefields({updated_pf_indicies})')
        
#     def btn_hide_all_callback(self, event):
#         print('btn_hide_all_callback(...)')
#         self.on_hide_all_placefields()
        
#     def btn_update_active_placefields(self, event):
#         print('btn_update_active_placefields(...)')
#         self.on_update_active_placefields(self.cross_selector.value)
        
#     def callback(self, *events):
#         print(events)
#         for event in events:
#             if event.name == 'options':
#                 self.selections.object = 'Possible options: %s' % ', '.join(event.new)
#             elif event.name == 'value':
#                 self.selected.object = 'Selected: %s' % ','.join(event.new)

#     def panel(self):
#         # Panel pane and widget objects:
#         self.selections = pn.pane.Markdown(object='')
#         self.selected = pn.pane.Markdown(object='')
#         self.cross_selector = pn.widgets.CrossSelector(name='Active Placefields', value=[], options=['0', '1', '2'], height=600, width=200) # cross_selector.value

#         # Action Buttons:
#         self.button_hide_all = pn.widgets.Button(name='Hide All Placefields')
#         self.button_hide_all.on_click(self.btn_hide_all_callback)
#         self.button_update = pn.widgets.Button(name='Update Active Placefields', button_type='primary')
#         self.button_update.on_click(self.btn_update_active_placefields)

#         self.watcher = self.cross_selector.param.watch(self.callback, ['options', 'value'], onlychanged=False)
#         # set initial
#         active_new_pf_panel.set_initial(self.num_pfs, [0, 1, 5])
        
#         return pn.Column(pn.Row(self.cross_selector, width=200, height=600),
#                          pn.Spacer(width=200, height=10),
#                          self.selections,
#                          pn.Spacer(width=200, height=10),
#                          self.selected,
#                          pn.Spacer(width=200, height=20),
#                          pn.Row(self.button_hide_all, self.button_update)
#                         )
    
#         # return pn.Row(pn.Column(self.cross_selector, width=200, height=600), self.selections, pn.Spacer(width=50, height=600), self.selected)

#     def set_initial(self, num_pfs, selected_values):
#         # set initial
#         # options = ['A','B','C','D']
#         # options = ['A','B','C','D']
#         pf_options_list_ints, pf_options_list_strings = ActivePlacefieldsPlotting.build_pf_options_list(num_pfs)
#         options = pf_options_list_strings
#         selected_values = [str(an_item) for an_item in selected_values]
#         # value=[]
#         # self.cross_selector.param.set_param(options=dict(zip(options,options)), value=['D'])
#         # self.cross_selector.param.set_param(options=dict(zip(options,options)), value=['D'])
#         self.cross_selector.param.set_param(options=dict(zip(options, options)), value=selected_values)
        