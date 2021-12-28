import param
import numpy as np
import pandas as pd
import pyvista as pv

from PhoPositionalData.analysis.helpers import partition
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
    spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
    
    @property
    def spikes_df(self):
        """The spikes_df property."""
        return self.active_session.spikes_df


    def plot_spikes(self):
        historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data_df(self.active_session.spikes_df, spike_geom=SpikeRenderingMixin.spike_geom_cone.copy())        
        self.plots_data['spikes_pf_active'] = {'historical_spikes_pdata':historical_spikes_pdata, 'historical_spikes_pc':historical_spikes_pc}
        if historical_spikes_pc.n_points >= 1:
            # self.plots['spikes_pf_active'] = self.p.add_mesh(historical_spikes_pc, name='spikes_pf_active', scalars='cellID', cmap=self.active_config.plotting_config.active_cells_listed_colormap, show_scalar_bar=False, lighting=True, render=False)
            self.plots['spikes_pf_active'] = self.p.add_mesh(historical_spikes_pc, name='spikes_pf_active', scalars='rgb', rgb=True, show_scalar_bar=False, lighting=True, render=False)
            needs_render = True
        else:
            # self.plots['spikes_pf_active'] = self.p.add_mesh(
            self.p.remove_actor(self.plots['spikes_pf_active'])
            needs_render = True
        return needs_render


    def update_spikes(self):
        """ Called to programmatically update the rendered spikes by replotting after changing their visibility/opacity/postion/etc """
        # full rebuild (to be safe):
        historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data_df(self.spikes_df, spike_geom=SpikeRenderingMixin.spike_geom_cone.copy())
        self.plots_data['spikes_pf_active'] = {'historical_spikes_pdata':historical_spikes_pdata, 'historical_spikes_pc':historical_spikes_pc}
        
        # Update just the values that could change:
        self.plots_data['spikes_pf_active']['historical_spikes_pdata']['render_opacity'] = self.spikes_df['render_opacity'].values
        # ?? Is this rebuild needed after updating the pdata to see the changes in the pc_data (which is what is actually plotted)???
        self.plots_data['spikes_pf_active']['historical_spikes_pc'] = self.plots_data['spikes_pf_active']['historical_spikes_pdata'].glyph(scale=False, geom=SpikeRenderingMixin.spike_geom_cone.copy()) 
        # spike_history_pdata['render_opacity'] = active_flat_df['render_opacity'].values
        
        if self.plots_data['spikes_pf_active']['historical_spikes_pc'].n_points >= 1:
            # self.plots['spikes_pf_active'] = self.p.add_mesh(self.plots_data['spikes_pf_active']['historical_spikes_pc'], name='spikes_pf_active', scalars='cellID', cmap=self.active_config.plotting_config.active_cells_listed_colormap, show_scalar_bar=False, lighting=True, render=False)
            # self.plots['spikes_pf_active'] = self.p.add_mesh(self.plots_data['spikes_pf_active']['historical_spikes_pc'], name='spikes_pf_active', scalars='cellID', cmap=self.active_config.plotting_config.active_cells_listed_colormap, opacity='render_opacity', show_scalar_bar=False, lighting=True, render=False)
            self.plots['spikes_pf_active'] = self.p.add_mesh(self.plots_data['spikes_pf_active']['historical_spikes_pc'], name='spikes_pf_active', scalars='rgb', rgb=True, show_scalar_bar=False, lighting=True, render=False)
            needs_render = True
        else:
            self.p.remove_actor(self.plots['spikes_pf_active'])
            needs_render = True

        if needs_render:
            self.p.render()
            
            
    def build_flat_color_data(self):
        # TODO: could also add in 'render_exclusion_mask'
        # RGB Version:
        self.params.flat_spike_colors_array = np.array([self.params.pf_colors[:-1, idx] for idx in self.spikes_df['cell_idx'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        print(f'SpikeRenderMixin.build_flat_color_data(): built rgb array from pf_colors, droppping the alpha components: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')

        # Add the split RGB columns to the DataFrame
        self.spikes_df[['R','G','B']] = self.params.flat_spike_colors_array
        
        # RGBA version:
        # self.params.flat_spike_colors_array = np.array([self.params.pf_colors[:, idx] for idx in self.spikes_df['cell_idx'].to_numpy()]) # np.shape(flat_spike_colors) # (77726, 4)
        # self.params.flat_spike_colors_array = np.array([pv.parse_color(spike_color_info.rgb_hex, opacity=spike_color_info.render_opacity) for spike_color_info in self.spikes_df[['rgb_hex', 'render_opacity']].itertuples()])
        # print(f'SpikeRenderMixin.build_flat_color_data(): built combined rgba array from rgb_hex and render_opacity: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')
        return self.params.flat_spike_colors_array
        
    
    # Testing:
    def test_toggle_cell_spikes_visibility(self, included_cell_ids):
        self.update_active_spikes(np.isin(self.spikes_df['aclu'], included_cell_ids))
      
        
    # Testing: Split spikes plot
    def build_active_unit_split_spikes_data(self, active_flat_df: pd.DataFrame):
        cell_split_df = partition(active_flat_df, 'aclu')
        for a_split_df in cell_split_df:
            spike_history_pdata, spike_history_pc = build_active_spikes_plot_data_df(a_split_df, SpikeRenderingMixin.spike_geom_cone.copy())
            # SingleCellSpikePlotData(point_data=spike_history_pdata, glyph_data=spike_history_pc)
            
    def setup_spike_rendering_mixin(self):
        # Add the required spike colors
        included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        flat_spike_hex_colors = np.array([self.params.pf_colors_hex[i] for i in self.spikes_df['cell_idx'].to_numpy()])
        self.spikes_df['rgb_hex'] = flat_spike_hex_colors.copy()
        self.build_flat_color_data()
        


class HideShowSpikeRenderingMixin:
    """ Implementors present spiking data with the option to hide/show/etc some of the outputs interactively. """    
    @property
    def spike_exclusion_mask(self):
        """The spike_exclusion_mask property."""
        return self.active_session.spikes_df['render_exclusion_mask']
    @spike_exclusion_mask.setter
    def spike_exclusion_mask(self, value):
        self.active_session.spikes_df['render_exclusion_mask'] = value    
    
    
    def setup_hide_show_spike_rendering_mixin(self):
        self.active_session.spikes_df['render_opacity'] = 0.0 # Initialize all spikes to 0.0 opacity, meaning they won't be rendered.
        self.active_session.spikes_df['render_exclusion_mask'] = False # all are included (not in the exclusion mask) to begin. This does not mean that they will be visible because 'render_opacity' is still set to zero.
        
        
    
    
    def update_active_spikes(self, spike_opacity_mask, is_additive=False):
        """ 
        Usage: 
            included_cell_ids = [48, 61]
            
            ipcDataExplorer.update_active_spikes(np.isin(ipcDataExplorer.active_session.spikes_df['aclu'], included_cell_ids)) # actives only the spikes that have aclu values (cell ids) in the included_cell_ids array.
        """
        assert np.shape(self.spikes_df['render_opacity']) == np.shape(spike_opacity_mask), "spike_opacity_mask must have one value for every spike in self.spikes_df, specifying its opacity"
        print(f'update_active_spikes(spike_opacity_mask: ..., is_additive: {is_additive})')
        if is_additive:
            # use the 'out' argument to re-assign it to the self.spikes_df['render_opacity'] array in-place:
            self.spikes_df['render_opacity'] = np.clip((self.spikes_df['render_opacity'] + spike_opacity_mask), 0.0, 1.0)
            # self.spikes_df['render_opacity'] = np.clip((self.spikes_df['render_opacity'] + spike_opacity_mask), 0.0, 1.0, self.spikes_df['render_opacity'])
        else:
            self.spikes_df['render_opacity'] = spike_opacity_mask
        self.update_spikes()
        
    # def include_unit_spikes(self, included_cell_ids):
    #     self.update_active_spikes(np.isin(self.spikes_df['aclu'], included_cell_ids), is_additive=True)
    
    
    
    def change_unit_spikes_included(self, cell_ids, are_included):
        """[summary]

        Args:
            cell_ids ([type]): [description]
            are_included ([type]): [description]
        """
        print(f'change_unit_spikes_included(cell_ids: {cell_ids}, are_included: {are_included})')
        if are_included:
            self.update_active_spikes(np.isin(self.spikes_df['aclu'], cell_ids), is_additive=True)
        else:
            # in remove mode, make the values negative:
            remove_opacity_specifier = np.isin(self.spikes_df['aclu'], cell_ids)
            remove_opacity = np.zeros(np.shape(remove_opacity_specifier))
            remove_opacity[remove_opacity_specifier] = -1 # set to negative one, to ensure that regardless of the current opacity the clipped opacity will be removed for these items
            self.update_active_spikes(remove_opacity, is_additive=True)
        
        
    def mask_spikes_from_render(self, excluded_cell_ids):
        self.spike_exclusion_mask[np.isin(self.spikes_df['aclu'], excluded_cell_ids)] = True
        self.update_spikes()
        
    def unmask_spikes_from_render(self, excluded_cell_ids):
        # removes the specified spikes from the exclusion mask
        self.spike_exclusion_mask[np.isin(self.spikes_df['aclu'], excluded_cell_ids)] = False
        self.update_spikes()
        
    def clear_spikes_exclusion_mask(self):
        self.spike_exclusion_mask = False # all are included (not in the exclusion mask) to begin.
        self.update_spikes()

    # def change_active_spikes_exclusion_mask(self, included_cell_ids, is_visible):        
    #     self.spikes_df['render_opacity']
        
    # # DEPRICATED:
    # def hide_placefield_spikes(self, active_original_cell_unit_ids, should_invert=True):
    #     # print('hide_placefield_spikes(active_index: {}, should_invert: {})'.format(active_original_cell_unit_ids, should_invert))
    #     mesh = self.plots_data['spikes_pf_active']['historical_spikes_pc'].cast_to_unstructured_grid()
    #     num_mesh_cells = mesh.n_cells
    #     ghosts = np.argwhere(np.isin(mesh["cellID"], active_original_cell_unit_ids, invert=should_invert))
    #     num_ghosts = len(ghosts)
    #     # print('\t num_mesh_cells: {}, num_ghosts: {}'.format(num_mesh_cells, num_ghosts))
    #     # This will act on the mesh inplace to mark those cell indices as ghosts
    #     mesh.remove_cells(ghosts)
    #     return mesh
