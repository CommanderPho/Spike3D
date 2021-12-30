# occupancy_plotting_mixins.py
# from PhoGui.InteractivePlotter.shared_helpers import PlotGroupWrapper
import param
from PhoPositionalData.plotting.graphs import plot_3d_binned_bars, plot_point_labels
from PhoPositionalData.plotting.mixins.general_plotting_mixins import BasePlotDataParams


class OccupancyPlottingConfig(BasePlotDataParams):
    debug_logging = False

    @staticmethod
    def _config_update_watch_labels():
        return ['barOpacity', 'labelsShowPoints', 'labelsOpacity', 'dropBelowThreshold']
    @staticmethod
    def _config_visibility_watch_labels():
        return ['labelsAreVisible', 'isVisible']
    
    # Overriding defaults from parent
    name = param.String(default='Occupancy')
    # name = param.Parameter(default='name', doc='The name of the placefield')
    isVisible = param.Boolean(default=True, doc="Whether the plot is visible")

    # Bar properties:
    barOpacity = param.Number(default=0.75, bounds=(0.0, 1.0), step=0.1)
    
    # Labels properties:
    labelsAreVisible = param.Boolean(default=False, doc="Whether the point labels are visible")
    labelsShowPoints = param.Boolean(default=False, doc="Whether to show the label's points on the figure")
    labelsOpacity = param.Number(default=0.5, bounds=(0.0, 1.0), step=0.1)

    # General properties:    
    dropBelowThreshold = param.Number(default=1E-6)
    
    def to_plot_config_dict(self):
        issue_labels = {'name': 'OccupancyLabels', 'name': 'Occupancy'}
        return {'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity, 'shape': 'rounded_rect', 'shape_opacity': self.labelsOpacity, 'show_points': self.labelsShowPoints}
    
    def to_bars_plot_config_dict(self):
        return {'name': 'Occupancy', 'drop_below_threshold': self.dropBelowThreshold, 'opacity': self.barOpacity}
    
    def to_labels_plot_config_dict(self):
        return {'name': 'OccupancyLabels', 'shape': 'rounded_rect', 'shape_opacity': self.labelsOpacity, 'show_points': self.labelsShowPoints}
    

    
class OccupancyPlottingMixin:
    """ Implementor visually plots a 3D occupancy map """
    debug_logging = False
    
    @property
    def ratemap(self):
        return self.params.active_epoch_placefields.ratemap
    
    @property
    def occupancy(self):
        return self.params.active_epoch_placefields.ratemap.occupancy
            
    # xbin & ybin properties  
    @property
    def xbin(self):
        return self.ratemap.xbin
    @property
    def ybin(self):
        return self.ratemap.ybin
    @property
    def xbin_centers(self):
        return self.ratemap.xbin_centers
    @property
    def ybin_centers(self):
        return self.ratemap.ybin_centers    

    @property
    def occupancy_plotting_config(self):
        """The occupancy_plotting_config property."""
        return self.params._active_occupancy_plotting_config
    @occupancy_plotting_config.setter
    def occupancy_plotting_config(self, value):
        self.params._active_occupancy_plotting_config = value

    @property
    def occupancy_plot_actor(self):
        return self.plots['occupancyPlotActor']
    
    # @property
    # def tuning_curve_is_visible(self):
    #     return np.array([bool(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors], dtype=bool)
        
    # @property
    # def tuning_curve_visibilities(self):
    #     return np.array([int(an_actor.GetVisibility()) for an_actor in self.tuning_curve_plot_actors], dtype=int)

    def plot_occupancy_bars(self, *arg):
        """Renders the internal occupancy map provided by self.occupancy as a series of bars spanning the xbin and ybins of the internal ratemap. """
        if self.debug_logging:
            print(f'OccupancyPlottingMixin.plot_occupancy_bars(): config: {self.occupancy_plotting_config}')
            
        self.plots['occupancyPlotActor'], self.plots_data['occupancyPlotData'] = self._perform_plot_occupancy_bars(self.occupancy_plotting_config.to_bars_plot_config_dict(), self.occupancy_plotting_config.to_labels_plot_config_dict())
        self.p.enable_depth_peeling() # this fixes bug where it appears transparent even when opacity is set to 1.00
        
    
    def on_occupancy_plot_config_updated(self, event):
        """ Called when the config is updated, such as by the panel GUI """
        # print(f'event_arg: {event_arg}')
        # event_arg.name
        # event_arg.new # the new value
        updated_config = event.obj
        if event.name == 'isVisible':
            # update the value in the internal config:
            self.occupancy_plotting_config.isVisible = updated_config.isVisible
        elif event.name == 'labelsAreVisible':
            self.occupancy_plotting_config.labelsAreVisible = updated_config.labelsAreVisible
        else:
            print(f'WARNING: OccupancyPlottingMixin.on_occupancy_plot_config_updated(event): Unknown event {event.name}: {event}')
            

    def on_occupancy_plot_update_visibility(self, *arg):
        # assert (self.occupancy_plot_actor is not None), "occupancy_plot_actor hasn't been initialized yet!"
        if (self.occupancy_plot_actor is None):
            # Auto-setup this mixin when this function is called if no others have been
            self.setup_occupancy_plotting_mixin()
            self.plot_occupancy_bars()
            # self.on_occupancy_plot_update_visibility() # call this to make sure the visibility is correct
        
        
        active_bars_actor_dict = self.occupancy_plot_actor.get('plot_3d_binned_bars_Occupancy', None)
        if active_bars_actor_dict is not None:
            active_bars_actor = active_bars_actor_dict.get('main', None)
            if active_bars_actor is not None:
                # has valid bars actor:
                # active_bars_actor = occupancy_plot_actor['plot_3d_binned_bars_Occupancy']['main']
                if self.occupancy_plotting_config.isVisible:
                    active_bars_actor.SetVisibility(1)
                else:
                    active_bars_actor.SetVisibility(0)   
            else:
                if self.occupancy_plotting_config.isVisible:
                    self.plot_occupancy_bars() # if the labels are supposed to be visible but don't exist on the plot yet, create them
                    
                     
        active_labels_actor = self.occupancy_plot_actor.get('plot_point_labels_OccupancyLabels', None)
        if active_labels_actor is not None:
            # has valid labels actor:
            if self.occupancy_plotting_config.labelsAreVisible:
                active_labels_actor.SetVisibility(1)
            else:
                active_labels_actor.SetVisibility(0)
        else:
            if self.occupancy_plotting_config.labelsAreVisible:
                self.plot_occupancy_bars() # if the labels are supposed to be visible but don't exist on the plot yet, create them
        

    def _perform_plot_occupancy_bars(self, bars_kwargs_dict=None, labels_kwargs_dict=None):
        """Renders the internal occupancy map provided by self.occupancy as a series of bars spanning the xbin and ybins of the internal ratemap.
        Usage:
            plotActors, data_dict = ipcDataExplorer.plot_occupancy_bars()
        """
        if bars_kwargs_dict is not None:        
            plotActors, data_dict = plot_3d_binned_bars(self.p, self.xbin, self.ybin, self.occupancy,
                                                **({'drop_below_threshold': 1e-06, 'name': 'Occupancy', 'opacity': 0.75, 'render': False} | bars_kwargs_dict))        
        # # The full point shown:
        # # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
        # # Only the z-values
        # point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'
        # # point_masking_function = lambda points: points[:, 2] > 20.0
        # point_masking_function = lambda points: points[:, 2] > 1E-6

        if (labels_kwargs_dict is not None) and self.occupancy_plotting_config.labelsAreVisible: 
            plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(self.p, self.xbin_centers, self.ybin_centers, self.occupancy, 
                                                                            **({'shape': 'rounded_rect', 'shape_opacity': 0.5, 'show_points': False, 'name': 'OccupancyLabels', 'render': True} | labels_kwargs_dict))
            # , **({'shape': 'rounded_rect', 'shape_opacity': 0.5, 'show_points': False, 'name': 'OccupancyLabels', 'render': True} | kwargs)
        else:
            plotActors_CenterLabels = dict()
            data_dict_CenterLabels = dict()
            
        plotActors = plotActors | plotActors_CenterLabels

        # print(f'plotActors: {plotActors}')        
        # print(f'plotActors_CenterLabels: {plotActors_CenterLabels}')
        
        # print(f'plotActors: {plotActors}')      
        # occupancy_bars_plotGroup = PlotGroupWrapper('3d_Occupancy_bars', {'plot_3d_binned_bars_Occupancy':plotActors['plot_3d_binned_bars_Occupancy']['main'], 'plot_point_labels_OccupancyLabels':plotActors['plot_point_labels_OccupancyLabels']})
        
        return plotActors, (data_dict | data_dict_CenterLabels)
    
    def setup_occupancy_plotting_mixin(self):
        self.occupancy_plotting_config = OccupancyPlottingConfig()
        self.plots['occupancyPlotActor'], self.plots_data['occupancyPlotData'] = None, None
        
        # Setup watchers:    
        self.occupancy_plotting_config.param.watch(self.plot_occupancy_bars, OccupancyPlottingConfig._config_update_watch_labels(), queued=True)
        self.occupancy_plotting_config.param.watch(self.on_occupancy_plot_update_visibility, OccupancyPlottingConfig._config_visibility_watch_labels(), queued=True)
    
  
    @classmethod
    def perform_plot_ratemap_bars(cls, p, ratemap,  point_labeling_function=None, point_masking_function=None):
        plotActors, data_dict = plot_3d_binned_bars(p, ratemap.xbin, ratemap.ybin, ratemap.occupancy, drop_below_threshold=1E-6, name='Occupancy', opacity=0.75)
        # , **({'drop_below_threshold': 1e-06, 'name': 'Occupancy', 'opacity': 0.75} | kwargs)

        if point_labeling_function is None:
            # The full point shown:
            # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
            # Only the z-values
            point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'

        if point_masking_function is None:
            # point_masking_function = lambda points: points[:, 2] > 20.0
            point_masking_function = lambda points: points[:, 2] > 1E-6

        plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(p, ratemap.xbin_centers, ratemap.ybin_centers, ratemap.occupancy, 
                                                                            point_labels=point_labeling_function, 
                                                                            point_mask=point_masking_function,
                                                                            shape='rounded_rect', shape_opacity= 0.5, show_points=False, name='OccupancyLabels')
        # , **({'point_labels': <function <lambda> at 0x0000018E817420D0>, 'point_mask': <function <lambda> at 0x0000018E8174F1F0>, 'shape': 'rounded_rect', 'shape_opacity': 0.5, 'show_points': False, 'name': 'OccupancyLabels'} | kwargs)
        
        return plotActors, data_dict
