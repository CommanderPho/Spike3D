import param
import numpy as np
import pandas as pd


class OptionsListMixin:
    @staticmethod
    def options_to_str(options_list_ints):
        return [f'{i}' for i in options_list_ints]
    @staticmethod
    def options_to_int(options_list_strings):
        return [int(a_str) for a_str in options_list_strings]
    @staticmethod
    def build_pf_options_list(num_pfs=40):
        pf_options_list_ints = np.arange(num_pfs)
        pf_options_list_strings = OptionsListMixin.options_to_str(pf_options_list_ints) # [f'{i}' for i in pf_options_list_ints]
        return pf_options_list_ints, pf_options_list_strings


    
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
    
    
# def __build_callbacks(self, tuningCurvePlotActors):
#         combined_active_pf_update_callbacks = []
#         for i, an_actor in enumerate(tuningCurvePlotActors):
#             # Make a separate callback for each widget
#             curr_visibility_callback = SetVisibilityCallback(an_actor)
#             curr_spikes_update_callback = (lambda is_visible, i_copy=i: self._update_placefield_spike_visibility([i_copy], is_visible))
#             combined_active_pf_update_callbacks.append(CallbackSequence([curr_visibility_callback, curr_spikes_update_callback]))
#         return combined_active_pf_update_callbacks
    
        


## Parameters (Param):
class BasePlotDataParams(param.Parameterized):
    # name = param.Parameter(default="Not editable", constant=True)
    name = param.String(default='name', doc='The name of the placefield')
    # name = param.Parameter(default='name', doc='The name of the placefield')
    isVisible = param.Boolean(default=False, doc="Whether the plot is visible")

class ExtendedPlotDataParams(BasePlotDataParams):
    color = param.Color(default='#FF0000', doc="The placefield's Color")
    extended_values_dictionary = param.Dict(default={}, doc="Extra values stored in a dictionary.")


class ExampleExtended(BasePlotDataParams):
    color                   = param.Color(default='#BBBBBB')
    dictionary              = param.Dict(default={"a": 2, "b": 9})
    select_string           = param.ObjectSelector(default="yellow", objects=["red", "yellow", "green"])
    select_fn               = param.ObjectSelector(default=list,objects=[list, set, dict])
    int_list                = param.ListSelector(default=[3, 5], objects=[1, 3, 5, 7, 9], precedence=0.5)

# checkbutton_group = pn.widgets.CheckButtonGroup(name='Check Button Group', value=[], options=pf_options_list_strings) # checkbutton_group.value 
# cross_selector = pn.widgets.CrossSelector(name='Active Placefields', value=[], options=pf_options_list_strings) # cross_selector.value


        


