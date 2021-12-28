import param
import numpy as np
import pandas as pd
import panel as pn


## TODO: this has been factored out and into neuropy.neuron_identities.NeuronIdentityAccessingMixin
class NeuronIdentityAccessingMixin:
    @property
    def neuron_ids(self):
        """ e.g. return np.array(active_epoch_placefields2D.cell_ids) """
        raise NotImplementedError
    
    def get_neuron_id_and_idx(self, neuron_i=None, neuron_id=None):
        assert (neuron_i is not None) or (neuron_id is not None), "You must specify either cell_i or cell_id, and the other will be returned"
        if neuron_i is not None:
            neuron_i = int(neuron_i)
            neuron_id = self.neuron_ids[neuron_i]
        elif neuron_id is not None:
            neuron_id = int(neuron_id)
            neuron_i = np.where(self.neuron_ids == neuron_id)[0].item()
        # print(f'cell_i: {cell_i}, cell_id: {cell_id}')
        return neuron_i, neuron_id
    
class HideShowSpikeRenderingMixin:
    def update_active_spikes(self, spike_opacity_mask):
        """ 
        Usage: 
            included_cell_ids = [48, 61]
            
            ipcDataExplorer.update_active_spikes(np.isin(ipcDataExplorer.active_session.spikes_df['aclu'], included_cell_ids)) # actives only the spikes that have aclu values (cell ids) in the included_cell_ids array.
        """
        assert np.shape(self.active_session.spikes_df['render_opacity']) == np.shape(spike_opacity_mask), "spike_opacity_mask must have one value for every spike in self.active_session.spikes_df, specifying its opacity"
        self.active_session.spikes_df['render_opacity'] = spike_opacity_mask
        self.update_spikes()
        
        
        
class HideShowPlacefieldsRenderingMixin(NeuronIdentityAccessingMixin, param.Parameterized):
    
    active_pf_idx_list = param.ListSelector(default=[3, 5], objects=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], precedence=0.5)
    # phase = param.Number(default=0, bounds=(0, np.pi))
    # frequency = param.Number(default=1, bounds=(0.1, 2))
    
    
    @param.depends('active_pf_idx_list')
    def interact_update_active_placefields(self):
        selected_placefield_indicies = self.active_pf_idx_list
        print(f'selected_placefield_indicies: {selected_placefield_indicies}')
        self.update_active_placefields(selected_placefield_indicies)
        
        
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
        for aTuningCurveActor in self.plots['tuningCurvePlotActors']:
            aTuningCurveActor.SetVisibility(0)
            
    def _show_tuning_curve(self, show_index):
        # Works to show the specified tuning curve plots:
        self.plots['tuningCurvePlotActors'][show_index].SetVisibility(1)


class BaseClass(param.Parameterized):
    name                       = param.Parameter(default="Not editable", constant=True)
    isVisible                 = param.Boolean(True, doc="Whether the plot is visible")
    


class ExampleExtended(BaseClass):
    color                   = param.Color(default='#BBBBBB')
    dictionary              = param.Dict(default={"a": 2, "b": 9})
    select_string           = param.ObjectSelector(default="yellow", objects=["red", "yellow", "green"])
    select_fn               = param.ObjectSelector(default=list,objects=[list, set, dict])
    int_list                = param.ListSelector(default=[3, 5], objects=[1, 3, 5, 7, 9], precedence=0.5)
    


class PlacefieldPlottingExtended(BaseClass):
    color                   = param.Color(default='#FF0000')
    dictionary              = param.Dict(default={"a": 2, "b": 9})
    select_string           = param.ObjectSelector(default="yellow", objects=["red", "yellow", "green"])
    select_fn               = param.ObjectSelector(default=list,objects=[list, set, dict])
    active_pf_idx_list                = param.ListSelector(default=[3, 5], objects=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], precedence=0.5)
    hide_all_action       = param.Action(lambda x: update_active_placefields(dt.datetime.utcnow()), 
                                           doc="""Hide All Placefields.""", precedence=0.7),
    update_action       = param.Action(lambda x: x.timestamps.append(dt.datetime.utcnow()), 
                                           doc="""Update Placefield Visibility.""", precedence=0.7),
    update_action       = param.Action(lambda x: x.timestamps.append(dt.datetime.utcnow()), 
                                           doc="""Update Placefield Visibility.""", precedence=0.7)



    



class ActivePlacefieldsPlottingNew:
    
    def __init__(self, num_pfs, **params):
        super(ActivePlacefieldsPlottingNew, self).__init__(**params)
        self.num_pfs = num_pfs
        # self.figure = figure(x_range=(-1, 1), y_range=(-1, 1))
        # self.renderer = self.figure.line(*self._get_coords())
    
    @staticmethod
    def build_pf_options_list(num_pfs=40):
        pf_options_list_ints = np.arange(num_pfs)
        pf_options_list_strings = [f'{i}' for i in pf_options_list_ints]
        return pf_options_list_ints, pf_options_list_strings
    
    # _on_hide_all_placefields = lambda x: print(f'_on_hide_all_placefields({x})')
    # _on_update_active_placefields = lambda x: print(f'_on_update_active_placefields({x})')
    
    def on_hide_all_placefields(self):
        print('on_hide_all_placefields()')
        # lambda x: print(f'_on_hide_all_placefields({x})')
    
    def on_update_active_placefields(self, updated_pf_indicies):
        print(f'on_update_active_placefields({updated_pf_indicies})')
        
    def btn_hide_all_callback(self, event):
        print('btn_hide_all_callback(...)')
        self.on_hide_all_placefields()
        
    def btn_update_active_placefields(self, event):
        print('btn_update_active_placefields(...)')
        self.on_update_active_placefields(self.cross_selector.value)
        
    def callback(self, *events):
        print(events)
        for event in events:
            if event.name == 'options':
                self.selections.object = 'Possible options: %s' % ', '.join(event.new)
            elif event.name == 'value':
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

        self.watcher = self.cross_selector.param.watch(self.callback, ['options', 'value'], onlychanged=False)
        # set initial
        active_new_pf_panel.set_initial(self.num_pfs, [0, 1, 5])
        
        return pn.Column(pn.Row(self.cross_selector, width=200, height=600),
                         pn.Spacer(width=200, height=10),
                         self.selections,
                         pn.Spacer(width=200, height=10),
                         self.selected,
                         pn.Spacer(width=200, height=20),
                         pn.Row(self.button_hide_all, self.button_update)
                        )
    
        # return pn.Row(pn.Column(self.cross_selector, width=200, height=600), self.selections, pn.Spacer(width=50, height=600), self.selected)

    def set_initial(self, num_pfs, selected_values):
        # set initial
        # options = ['A','B','C','D']
        # options = ['A','B','C','D']
        pf_options_list_ints, pf_options_list_strings = ActivePlacefieldsPlotting.build_pf_options_list(num_pfs)
        options = pf_options_list_strings
        selected_values = [str(an_item) for an_item in selected_values]
        # value=[]
        # self.cross_selector.param.set_param(options=dict(zip(options,options)), value=['D'])
        # self.cross_selector.param.set_param(options=dict(zip(options,options)), value=['D'])
        self.cross_selector.param.set_param(options=dict(zip(options, options)), value=selected_values)
        