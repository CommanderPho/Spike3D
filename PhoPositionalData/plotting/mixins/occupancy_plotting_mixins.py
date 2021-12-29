# occupancy_plotting_mixins.py
# from PhoGui.InteractivePlotter.shared_helpers import PlotGroupWrapper
from PhoPositionalData.plotting.graphs import plot_3d_binned_bars, plot_point_labels

class OccupancyPlottingMixin:
    """ Implementor visually plots a 3D occupancy map """
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


    def plot_occupancy_bars(self):
        """Renders the internal occupancy map provided by self.occupancy as a series of bars spanning the xbin and ybins of the internal ratemap.
        Usage:
            plotActors, data_dict = ipcDataExplorer.plot_occupancy_bars()
        """
        plotActors, data_dict = plot_3d_binned_bars(self.p, self.xbin, self.ybin, self.occupancy,
                                                drop_below_threshold=1E-6, name='Occupancy', opacity=0.75, render=False)

        # The full point shown:
        # point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
        # Only the z-values
        point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'
        # point_masking_function = lambda points: points[:, 2] > 20.0
        point_masking_function = lambda points: points[:, 2] > 1E-6

        plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(self.p, self.xbin_centers, self.ybin_centers, self.occupancy, 
                                                                            point_labels=point_labeling_function, 
                                                                            point_mask=point_masking_function,
                                                                            shape='rounded_rect', shape_opacity= 0.5, show_points=False, name='OccupancyLabels', render=True)

        # print(f'plotActors: {plotActors}')        
        # print(f'plotActors_CenterLabels: {plotActors_CenterLabels}')
        plotActors = plotActors | plotActors_CenterLabels
        # print(f'plotActors: {plotActors}')      
        # occupancy_bars_plotGroup = PlotGroupWrapper('3d_Occupancy_bars', {'plot_3d_binned_bars_Occupancy':plotActors['plot_3d_binned_bars_Occupancy']['main'], 'plot_point_labels_OccupancyLabels':plotActors['plot_point_labels_OccupancyLabels']})
        return plotActors, (data_dict | data_dict_CenterLabels)
    
  
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
