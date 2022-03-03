from typing import OrderedDict
import param
import numpy as np
import pandas as pd
import pyvista as pv

from PhoPositionalData.analysis.helpers import partition
from PhoPositionalData.plotting.mixins.general_plotting_mixins import NeuronConfigOwningMixin
from PhoPositionalData.plotting.spikeAndPositions import build_active_spikes_plot_data_df

from pyphocorehelpers.indexing_helpers import safe_get

# class SingleCellSpikePlotData(param.Parameterized):
#     point_data = param.Array(doc='spike_history_pdata')
#     glyph_data = param.Array(doc='spike_history_pc')


# class SpikePlotData(param.Parameterized):
#     plot_data = SingleCellSpikePlotData.param

class SpikesDataframeOwningMixin:
    """ Implementors own a spikes_df object """
    @property
    def spikes_df(self):
        """The spikes_df property."""
        return self.active_session.spikes_df


    def find_rows_matching_cell_IDXs(self, cell_IDXs):
        """Finds the cell IDXs (not IDs) in the self.spikes_df's appropriate column
        Args:
            cell_IDXs ([type]): [description]
        """
        return np.isin(self.spikes_df['cell_idx'], cell_IDXs)
    
    def find_rows_matching_cell_ids(self, cell_ids):
        """Finds the cell original ID in the self.spikes_df's appropriate column
        Args:
            cell_ids ([type]): [description]
        """
        return np.isin(self.spikes_df['aclu'], cell_ids)


# Typically requires conformance to SpikesDataframeOwningMixin
class SpikeRenderingMixin:
    """ Implementors render spikes from neural data in 3D 
        Requires:
            self.spikes_df
            self.find_rows_matching_cell_IDXs(self, cell_IDXs)
            self.find_rows_matching_cell_ids(self, cell_ids)
    """
    debug_logging = False
    spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
    
    def plot_spikes(self):
        historical_spikes_pdata, historical_spikes_pc = build_active_spikes_plot_data_df(self.spikes_df, spike_geom=SpikeRenderingMixin.spike_geom_cone.copy())        
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

    # Testing:
    def test_toggle_cell_spikes_visibility(self, included_cell_ids):
        self.update_active_spikes(np.isin(self.spikes_df['aclu'], included_cell_ids))
      
        
    # Testing: Split spikes plot
    def build_active_unit_split_spikes_data(self, active_flat_df: pd.DataFrame):
        """ 
        TODO: Unused. 
        """
        cell_split_df = partition(active_flat_df, 'aclu')
        for a_split_df in cell_split_df:
            spike_history_pdata, spike_history_pc = build_active_spikes_plot_data_df(a_split_df, SpikeRenderingMixin.spike_geom_cone.copy())
            # SingleCellSpikePlotData(point_data=spike_history_pdata, glyph_data=spike_history_pc)
      
                  
            
    def _build_flat_color_data(self, fallback_color_rgba = (0, 0, 0, 1.0)):
        """ Called only by self.setup_spike_rendering_mixin()
        
        # Adds to self.params:
            opaque_pf_colors
            
            flat_spike_colors_array # for some reason. Length of spikes_df
            
            cell_spike_colors_dict
            cell_spike_opaque_colors_dict
        
        # Adds columns to self.spikes_df:
            'rgb_hex','R','G','B'
        
        """
        # adds the color information to the self.spikes_df using params.pf_colors. Adds ['R','G','B'] columns and creates a self.params.flat_spike_colors_array with one color for each spike.
        # fallback_color_rgb: the default value to use for colors that aren't present in the pf_colors array
        fallback_color_rgb = fallback_color_rgba[:-1] # Drop the opacity component, so we only have RGB values
        
        # TODO: could also add in 'render_exclusion_mask'
        # RGB Version:
        self.params.opaque_pf_colors = self.params.pf_colors[:-1, :].copy() # Drop the opacity component, so we only have RGB values
        
        # Build flat hex colors, creating the self.spikes_df['rgb_hex'] column:
        flat_spike_hex_colors = np.array([safe_get(self.params.pf_colors_hex, cell_IDX, '#000000') for cell_IDX in self.spikes_df['cell_idx'].to_numpy()])        
        # flat_spike_hex_colors = np.array([self.params.pf_colors_hex[cell_IDX] for cell_IDX in self.spikes_df['cell_idx'].to_numpy()])
        self.spikes_df['rgb_hex'] = flat_spike_hex_colors.copy()

        # if type(self.params.pf_colors is np.array):
        unique_cell_indicies = np.unique(self.spikes_df['cell_idx'].to_numpy())
        max_cell_idx = np.max(unique_cell_indicies)
        num_unique_spikes_df_cell_indicies = len(unique_cell_indicies)
        
        # generate a dict of colors with an entry
        # pf_colors_dict = {cell_IDX: fallback_color_rgba for cell_IDX in unique_cell_indicies}
        # pf_opaque_colors_dict = {cell_IDX: fallback_color_rgb for cell_IDX in unique_cell_indicies}

        # Flat version:
        self.params.cell_spike_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgba]))
        self.params.cell_spike_opaque_colors_dict = OrderedDict(zip(unique_cell_indicies, num_unique_spikes_df_cell_indicies*[fallback_color_rgb]))
        
        num_pf_colors = np.shape(self.params.pf_colors)[0]
        valid_pf_colors_indicies = np.arange(num_pf_colors)
        for cell_IDX in unique_cell_indicies:
            if cell_IDX in valid_pf_colors_indicies:
                # if we have a color for it, use it
                self.params.cell_spike_colors_dict[cell_IDX] = self.params.pf_colors[:, cell_IDX]
                self.params.cell_spike_opaque_colors_dict[cell_IDX] = self.params.opaque_pf_colors[:, cell_IDX]
            else:
                # Otherwise use the fallbacks:
                self.params.cell_spike_colors_dict[cell_IDX] = fallback_color_rgba
                self.params.cell_spike_opaque_colors_dict[cell_IDX] = fallback_color_rgb
        
        # self.params.flat_spike_colors_array = np.array([safe_get(self.params.opaque_pf_colors, idx, fallback_color) for idx in self.spikes_df['cell_idx'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
        self.params.flat_spike_colors_array = np.array([self.params.cell_spike_opaque_colors_dict.get(idx, fallback_color_rgb) for idx in self.spikes_df['cell_idx'].to_numpy()]) # Drop the opacity component, so we only have RGB values. np.shape(flat_spike_colors) # (77726, 3)
        
        if self.debug_logging:
            print(f'SpikeRenderMixin.build_flat_color_data(): built rgb array from pf_colors, droppping the alpha components: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')
        # Add the split RGB columns to the DataFrame
        self.spikes_df[['R','G','B']] = self.params.flat_spike_colors_array
        # RGBA version:
        # self.params.flat_spike_colors_array = np.array([self.params.pf_colors[:, idx] for idx in self.spikes_df['cell_idx'].to_numpy()]) # np.shape(flat_spike_colors) # (77726, 4)
        # self.params.flat_spike_colors_array = np.array([pv.parse_color(spike_color_info.rgb_hex, opacity=spike_color_info.render_opacity) for spike_color_info in self.spikes_df[['rgb_hex', 'render_opacity']].itertuples()])
        # print(f'SpikeRenderMixin.build_flat_color_data(): built combined rgba array from rgb_hex and render_opacity: np.shape(self.params.flat_spike_colors_array): {np.shape(self.params.flat_spike_colors_array)}')
        return self.params.flat_spike_colors_array
              
    def setup_spike_rendering_mixin(self):
        """ Add the required spike colors built from the self.pf_colors. Spikes that do not contribute to a cell with a placefield are assigned a black color by default
        By Calling self._build_flat_color_data():
            # Adds to self.params:
                opaque_pf_colors
                
                flat_spike_colors_array # for some reason. Length of spikes_df
                
                cell_spike_colors_dict
                cell_spike_opaque_colors_dict
            
            # Adds columns to self.spikes_df:
                'cell_idx', 'rgb_hex','R','G','B'
            
        """
        included_cell_INDEXES = np.array([self.get_neuron_id_and_idx(neuron_id=an_included_cell_ID)[0] for an_included_cell_ID in self.spikes_df['aclu'].to_numpy()]) # get the indexes from the cellIDs
        self.spikes_df['cell_idx'] = included_cell_INDEXES.copy()
        # flat_spike_hex_colors = np.array(flat_spike_hex_colors)
        
        
        self._build_flat_color_data()
        
    
    


class HideShowSpikeRenderingMixin:
    """ Implementors present spiking data with the option to hide/show/etc some of the outputs interactively. """    
    debug_logging = False
        
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
        """ Main update callback function for visual changes. Updates the self.spikes_df.
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
            
    def change_unit_spikes_included(self, cell_IDXs=None, cell_IDs=None, are_included=True):
        """ Called to update the set of visible spikes for specified cell indicies or IDs
        Args:
            cell_ids ([type]): [description]
            are_included ([type]): [description]
        """
        assert (cell_IDXs is not None) or (cell_IDs is not None), "You must specify either cell_IDXs or cell_IDs, but not both"
        # TODO: could use the NeuronIdentityAccessingMixin helper class
            # self.get_neuron_id_and_idx(neuron_i=cell_IDXs, cell_ids=cell_ids)
        if cell_IDXs is not None:
            # IDXs mode, preferred.
            if self.debug_logging:
                print(f'HideShowSpikeRenderingMixin.change_unit_spikes_included(cell_IDXs: {cell_IDXs}, are_included: {are_included}): (note use of Index mode)')            
            matching_rows = self.find_rows_matching_cell_IDXs(cell_IDXs)
        else:
            # IDs mode.
            if self.debug_logging:
                print(f'HideShowSpikeRenderingMixin.change_unit_spikes_included(cell_IDs: {cell_IDs}, are_included: {are_included}): WARNING: cell_ID mode. Indexes are preferred.')
            # convert cell_IDs to to cell_IDXs for use later in updating the configs
            cell_IDXs = self.find_cell_IDXs_from_cell_ids(cell_IDs)
            matching_rows = self.find_rows_matching_cell_ids(cell_IDs)

        # Update the specific rows:
        self.change_spike_rows_included(matching_rows, are_included)
        
        # update the configs for these changed neurons:
        assert hasattr(self, 'update_neuron_render_configs'), "self must be of type NeuronConfigOwningMixin to have access to its configs"
        updated_configs = []
        for an_updated_config_idx in cell_IDXs:
            self.active_neuron_render_configs[an_updated_config_idx].spikesVisible = are_included # update the config
            updated_configs.append(self.active_neuron_render_configs[an_updated_config_idx])
        # call the parent (NeuronConfigOwningMixin) function to ensure the configs are updated.
        self.update_neuron_render_configs(cell_IDXs, updated_configs) # update configs
        

    def clear_all_spikes_included(self):
        # removes all spikes from inclusion
        if self.debug_logging:
            print(f'HideShowSpikeRenderingMixin.clear_spikes_included(): clearing all spikes.')     
        self.change_unit_spikes_included(cell_IDXs=self.neuron_config_indicies, are_included=False) # get all indicies, and set them all to excluded
           

    def change_spike_rows_included(self, row_specifier_mask, are_included):
        """change_spike_rows_included presents an IDX vs. ID agnostic interface with the self.spikes_df to allow the bulk of the code to work for both cases.

        Args:
            row_specifier_mask ([type]): the boolean mask indentifying rows of the dataframe.
            are_included ([type]): [description]
        """
        if are_included:
            self.update_active_spikes(row_specifier_mask, is_additive=True)
        else:
            # in remove mode, make the values negative:
            remove_opacity_specifier = row_specifier_mask # gets the only spikes that are included in the cell_ids
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
