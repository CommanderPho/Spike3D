# InteractivePyvistaPlotterBuildIfNeededMixin

# from pyvistaqt import BackgroundPlotter

# from neuropy
import numpy as np
import pyvista as pv
from pyvista.plotting.plotting import Plotter
# from pyvista.core.composite import MultiBlock
from pyvistaqt import BackgroundPlotter
from pyvistaqt.plotting import MultiPlotter

from pyphocorehelpers.DataStructure.general_parameter_containers import DebugHelper, VisualizationParameters

from pyphoplacecellanalysis.PhoPositionalData.plotting.gui import customize_default_pyvista_theme, print_controls_helper_text
from pyphoplacecellanalysis.PhoPositionalData.import_data import build_spike_positions_list


from pyphoplacecellanalysis.PhoPositionalData.plotting.spikeAndPositions import animal_location_circle, animal_location_trail_circle

# class PlotGroup:
#     """
#         # can plot all at once using:
#         blocks.plot()
            
#         for name in blocks.keys():
#             block = blocks[name]

#         for block in blocks:
#             surf = block.extract_surface()  # Do something with each dataset
#     """
#     def __init__(self, name, plots):
#         self.name = name
#         self.plots = plots
#         self.blocks = pv.MultiBlock(self.plots)

        
#         # # Make a tree.
#         # root = vtkMultiBlockDataSet()
#         # # make the default branch:
#         # branch = vtkMultiBlockDataSet()        
#         # root.SetBlock(0, branch)
        
#         # # apply the list objects as leaves
#         # for i, a_plot in enumerate(plots_list):
#         #     # Make some leaves.
#         #     a_leaf = a_plot
#         #     a_leaf.SetCenter(0, 0, 0)
#         #     a_leaf.Update()
#         #     branch.SetBlock(0, a_leaf.GetOutput())
        





class InteractivePyvistaPlotterBuildIfNeededMixin:
    @staticmethod
    def build_new_plotter_if_needed(pActiveTuningCurvesPlotter=None, plotter_type='BackgroundPlotter', **kwargs):
        """[summary]

        Args:
            pActiveTuningCurvesPlotter ([type], optional): [description]. Defaults to None.
            plotter_type (str, optional): [description]. ['BackgroundPlotter', 'MultiPlotter'] Defaults to 'BackgroundPlotter'.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if (pActiveTuningCurvesPlotter is not None):
            if isinstance(pActiveTuningCurvesPlotter, BackgroundPlotter):
                if pActiveTuningCurvesPlotter.app_window.isHidden():
                    print('No open BackgroundPlotter')
                    pActiveTuningCurvesPlotter.close() # Close it to start over fresh
                    pActiveTuningCurvesPlotter = None
                    needs_create_new_backgroundPlotter = True
                else:
                    print('BackgroundPlotter already open, reusing it.. NOT Forcing creation of a new one!')
                    pActiveTuningCurvesPlotter.close() # Close it to start over fresh
                    pActiveTuningCurvesPlotter = None
                    needs_create_new_backgroundPlotter = True
                    
            else:
                print(f'No open BackgroundPlotter, p is a {type(pActiveTuningCurvesPlotter)} object')
                pActiveTuningCurvesPlotter.close()
                pActiveTuningCurvesPlotter = None
                needs_create_new_backgroundPlotter = True
        else:
            print('No extant BackgroundPlotter')
            needs_create_new_backgroundPlotter = True

        if needs_create_new_backgroundPlotter:
            print(f'Creating a new {plotter_type}')
            # pActiveTuningCurvesPlotter = BackgroundPlotter(window_size=(1920, 1080), shape=(1,1), off_screen=False) # Use just like you would a pv.Plotter() instance
            if plotter_type == 'BackgroundPlotter':
                pActiveTuningCurvesPlotter = BackgroundPlotter(**({'window_size':(1920, 1080), 'shape':(1,1), 'off_screen':False} | kwargs)) # Use just like you would a pv.Plotter() instance 
            elif plotter_type == 'MultiPlotter':
                pActiveTuningCurvesPlotter = MultiPlotter(**({'window_size':(1920, 1080), 'shape':(1,1), 'off_screen':False} | kwargs))
            else:
                print(f'plotter_type is of unknown type {plotter_type}')
                raise ValueError
                
        return pActiveTuningCurvesPlotter


class InteractivePyvistaPlotter_ObjectManipulationMixin:
    """ Has a self.plots dict that uses string keys to access named plots
        This mixin adds functions that enables interactive manipulation of plotted objects post-hoc
    """
    ## Plot Manipulation Helpers:
    @property
    def get_plot_objects_list(self):
        """ a list of all valid plot objects """
        return list(self.plots.keys())

    @staticmethod
    def __toggle_visibility(mesh):
        new_vis = not bool(mesh.GetVisibility())
        mesh.SetVisibility(new_vis)
        # return new_vis

    def safe_get_plot(self, plot_key):
        a_plot = self.plots.get(plot_key, None)
        if a_plot is not None:
            return a_plot
        else:
            raise IndexError

    def set_plot_visibility(self, plot_key, is_visibie):
        self.safe_get_plot(plot_key).SetVisibility(is_visibie)

    def toggle_plot_visibility(self, plot_key):
        return InteractivePyvistaPlotter_ObjectManipulationMixin.__toggle_visibility(self.safe_get_plot(plot_key))


# ### UNUSED:
# class PlotGroupWrapper(InteractivePyvistaPlotter_ObjectManipulationMixin):
    
#     def __init__(self, name, plots_dict=dict(), gui_dict=dict()):
#         self.name = name
#         self.plots = plots_dict
#         # self.plots_data = dict()
#         self.gui_dict = gui_dict
        

#     def GetVisibility(self):
#         item_visibilities = np.array([a_plot.GetVisibility() for a_plot in self.plots.values()], dtype=int)
#         return np.any(item_visibilities)

#     def SetVisibility(self, value):
#         for a_plot_name, a_plot in self.plots.items():
#             a_plot.SetVisibility(value)

        
        
        

class InteractiveDataExplorerBase(InteractivePyvistaPlotterBuildIfNeededMixin, InteractivePyvistaPlotter_ObjectManipulationMixin):
    """The common base class for building an interactive PyVistaQT BackgroundPlotter with extra GUI components and controls.
    """
    def __init__(self, active_config, active_session, extant_plotter=None, data_explorer_name='InteractiveDataExplorerBase'):
        # active_session: Neuropy.core.Session
        self.active_config = active_config
        self.active_session = active_session
        self.p = extant_plotter
        self.data_explorer_name = data_explorer_name
        
        self.z_fixed = None
        # Position variables: t, x, y
        self.t = self.active_session.position.time
        self.x = self.active_session.position.x
        self.y = self.active_session.position.y
        
        # Helper variables
        self.params = VisualizationParameters('')
        self.debug = DebugHelper('')

        self.plots_data = dict()
        self.plots = dict()
        self.gui = dict()
        
    @staticmethod
    def _unpack_variables(active_session):
        # Spike variables: num_cells, spike_list, cell_ids, flattened_spikes
        num_cells = active_session.neurons.n_neurons
        spike_list = active_session.neurons.spiketrains
        cell_ids = active_session.neurons.neuron_ids
        # Gets the flattened spikes, sorted in ascending timestamp for all cells. Returns a FlattenedSpiketrains object
        flattened_spike_identities = np.concatenate([np.full((active_session.neurons.n_spikes[i],), active_session.neurons.neuron_ids[i]) for i in np.arange(active_session.neurons.n_neurons)]) # repeat the neuron_id for each spike that belongs to that neuron
        flattened_spike_times = np.concatenate(active_session.neurons.spiketrains)
        # Get the indicies required to sort the flattened_spike_times
        flattened_sort_indicies = np.argsort(flattened_spike_times)
        t_start = active_session.neurons.t_start
        reverse_cellID_idx_lookup_map = active_session.neurons.reverse_cellID_index_map

        # Position variables: t, x, y
        t = active_session.position.time
        x = active_session.position.x
        y = active_session.position.y
        linear_pos = active_session.position.linear_pos
        speeds = active_session.position.speed


        ### Build the flattened spike positions list
        # Determine the x and y positions each spike occured for each cell
        ## new_df style:
        flattened_spike_positions_list_new = active_session.flattened_spiketrains.spikes_df[["x", "y"]].to_numpy().T
        # print('\n flattened_spike_positions_list_new: {}, {}'.format(np.shape(flattened_spike_positions_list_new), flattened_spike_positions_list_new))
        # flattened_spike_positions_list_new: (2, 17449), [[ nan 0.37450201 0.37450201 ... 0.86633532 0.86632449 0.86632266], [ nan 0.33842111 0.33842111 ... 0.47504852 0.47503917 0.47503759]]
        flattened_spike_positions_list_new

        ## old-style:
        spike_positions_list = build_spike_positions_list(spike_list, t, x, y)
        flattened_spike_positions_list = np.concatenate(tuple(spike_positions_list), axis=1) # needs tuple(...) to conver the list into a tuple, which is the format it expects
        flattened_spike_positions_list = flattened_spike_positions_list[:, flattened_sort_indicies] # ensure the positions are ordered the same as the other flattened items so they line up
        # print('\n flattened_spike_positions_list_old: {}, {}\n\n'.format(np.shape(flattened_spike_positions_list), flattened_spike_positions_list))
        #  flattened_spike_positions_list_old: (2, 17449), [[103.53295196 100.94485182 100.86902972 ... 210.99778204 210.87296572 210.85173243]

        return num_cells, spike_list, cell_ids, flattened_spike_identities, flattened_spike_times, flattened_sort_indicies, t_start, reverse_cellID_idx_lookup_map, t, x, y, linear_pos, speeds, flattened_spike_positions_list


    def _setup(self):
        self._setup_variables()
        self._setup_visualization()
        self._setup_pyvista_theme()

    def _setup_variables(self):
        raise NotImplementedError
   
    def _setup_visualization(self):
        raise NotImplementedError

    def _setup_pyvista_theme(self):
        customize_default_pyvista_theme() # Sets the default theme values to those specified in my imported file
        # This defines the position of the vertical/horizontal splitting, in this case 40% of the vertical/horizontal dimension of the window
        # pv.global_theme.multi_rendering_splitting_position = 0.40
        pv.global_theme.multi_rendering_splitting_position = 0.80
        
    
    def plot(self, pActivePlotter=None):
        raise NotImplementedError
    
    def perform_plot_location_point(self, plot_name, curr_animal_point, render=True, **kwargs):
        """ will render a flat indicator of a single point like is used for the animal's current location. 
        Updates the existing plot if the same plot_name is reused. """
        ## COMPAT: merge operator '|'requires Python 3.9
        pdata_current_point = pv.PolyData(curr_animal_point) # a mesh
        pc_current_point = pdata_current_point.glyph(scale=False, geom=animal_location_circle)
        
        self.plots_data[plot_name] = {'pdata_current_point':pdata_current_point, 'pc_current_point':pc_current_point}
        self.plots[plot_name] = self.p.add_mesh(pc_current_point, name=plot_name, render=render, **({'color':'green', 'ambient':0.6, 'opacity':0.5,
                        'show_edges':True, 'edge_color':[0.05, 0.8, 0.08], 'line_width':3.0, 'nan_opacity':0.0, 'render_lines_as_tubes':True,
                        'show_scalar_bar':False, 'use_transparency':True} | kwargs))
        return self.plots[plot_name], self.plots_data[plot_name]


    def perform_plot_location_trail(self, plot_name, arr_x, arr_y, arr_z, render=True, trail_fade_values=None, trail_point_size_values=None, **kwargs):
        """ will render a series of points as a trajectory/path given arr_x, arr_y, and arr_z vectors of the same length.
        indicator of a single point like is used for the animal's current location. 
        Updates the existing plot if the same plot_name is reused. """
        point_cloud_fixedSegements_positionTrail = np.column_stack((arr_x, arr_y, arr_z))
        pdata_positionTrail = pv.PolyData(point_cloud_fixedSegements_positionTrail.copy()) # a mesh
        if trail_fade_values is not None:
            pdata_positionTrail.point_data['pho_fade_values'] = trail_fade_values
            scalars_arg = 'pho_fade_values'
        else:
            scalars_arg = None
        if trail_point_size_values is not None:
            pdata_positionTrail.point_data['pho_size_values'] = trail_point_size_values
            point_size_scale_arg = 'pho_size_values'
        else:
            point_size_scale_arg = None
        
        # create many spheres from the point cloud
        pc_positionTrail = pdata_positionTrail.glyph(scale=point_size_scale_arg, geom=animal_location_trail_circle)
        self.plots_data[plot_name] = {'point_cloud_fixedSegements_positionTrail':point_cloud_fixedSegements_positionTrail, 'pdata_positionTrail':pdata_positionTrail, 'pc_positionTrail':pc_positionTrail}
        self.plots[plot_name] = self.p.add_mesh(pc_positionTrail, name=plot_name, render=render, **({'ambient':0.6, 'opacity':'linear_r', 'scalars':scalars_arg, 'nan_opacity':0.0,
                                                'show_edges':False, 'render_lines_as_tubes':True, 'show_scalar_bar':False, 'use_transparency':True} | kwargs))
        return self.plots[plot_name], self.plots_data[plot_name]
            
            
            
