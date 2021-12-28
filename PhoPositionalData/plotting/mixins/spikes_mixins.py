import param
import numpy as np
import pandas as pd

from PhoPositionalData.analysis.helpers import partition
from PhoPositionalData.plotting.mixins.general_plotting_mixins import ExtendedPlotDataParams
from PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data_df


class SingleCellSpikePlotData(param.Parameterized):
    point_data = param.Array(doc='spike_history_pdata')
    glyph_data = param.Array(doc='spike_history_pc')


class SpikePlotData(param.Parameterized):
	plot_data = SingleCellSpikePlotData.param


# class SpikePlotData(ExtendedPlotDataParams):   
#     point_data = param.Array(doc='spike_history_pdata')
#     # @param.output(
#     glyph_data = param.Array(doc='spike_history_pc')
    
    
    
    

class SpikeRenderingMixin:
    def plot_spikes(self):
        pass
    
    def build_active_unit_split_spikes_data(self, active_flat_df: pd.DataFrame):
        cell_split_df = partition(active_flat_df, 'aclu')
        for a_split_df in cell_split_df:
            spike_history_pdata, spike_history_pc = build_active_spikes_plot_data_df(a_split_df, spike_geom.copy())
			SingleCellSpikePlotData(point_data=spike_history_pdata, glyph_data=spike_history_pc)
            



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
        
        