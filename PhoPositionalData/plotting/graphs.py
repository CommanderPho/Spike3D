from copy import deepcopy
import numpy as np
import pyvista as pv

def build_3d_plot_identifier_name(*args):
    return '_'.join(list(args))    


## 3D Binned Bar Plots:

def prepare_binned_data_for_3d_bars(xbin, ybin, data, mask2d=None):
    """ Sequentally repeats xbin, ybin, and data entries to prepare for being plot in 3D bar-plot form.
    Does this by repeating the xbin and ybin except the first and last entries so that there is one entry for each vertex of a 2d rectangular polygon.
    Following that, it repeats in both dimensions the data values, so that each of the created verticies has the same height value.
    Usage:
        modified_xbin, modified_ybin, modified_data = prepare_binned_data_for_3d_bars(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    # duplicates every xbin value except for the first and last:
    modified_xbin = np.repeat(xbin, 2) # remove the first and last elements, which are duplicates
    modified_xbin = modified_xbin[1:-1] 
    modified_ybin = np.repeat(ybin, 2) # remove the first and last elements, which are duplicates
    modified_ybin = modified_ybin[1:-1]
    # there should be precisely double the number of bins in each direction as there are data
    # print(f'np.shape(data): {np.shape(data)}, np.shape(modified_xbin): {np.shape(modified_xbin)}, np.shape(modified_ybin): {np.shape(modified_ybin)}')
    assert (np.shape(data)[0] * 2) == np.shape(modified_xbin)[0], "There shoud be double the number of xbins in the modified array as there are data points in the array."
    assert (np.shape(data)[1] * 2) == np.shape(modified_ybin)[0], "There shoud be double the number of ybins in the modified array as there are data points in the array."
    
    modified_data = np.repeat(data, 2, axis=0)
    modified_data = np.repeat(modified_data, 2, axis=1)
    
    if mask2d is not None:
        modified_mask2d = np.repeat(mask2d, 2, axis=0)
        modified_mask2d = np.repeat(modified_mask2d, 2, axis=1)
    else:
        modified_mask2d = None
    
    return modified_xbin, modified_ybin, modified_data, modified_mask2d
    
def plot_3d_binned_bars(p, xbin, ybin, data, zScalingFactor=1.0, drop_below_threshold: float=None, **kwargs):
    """ Plots a 3D bar-graph
    Usage:
        plotActors, data_dict = plot_3d_binned_data(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    if drop_below_threshold is not None:
        # print(f'drop_below_threshold: {drop_below_threshold}')
        # active_data[np.where(active_data < drop_below_threshold)] = np.nan
        data_mask = (data.copy() < drop_below_threshold)
    else:
        data_mask = None
    
    modified_xbin, modified_ybin, modified_data, modified_mask2d = prepare_binned_data_for_3d_bars(xbin.copy(), ybin.copy(), data.copy(), mask2d=data_mask)
    # build a structured grid out of the bins
    twoDimGrid_x, twoDimGrid_y = np.meshgrid(modified_xbin, modified_ybin)
    active_data = deepcopy(modified_data[:,:].T) # A single tuning curve
    # active_data = modified_data[:,:].copy() # A single tuning curve

    if modified_mask2d is not None:
        active_data_mask = modified_mask2d[:,:].T.copy()
        # print(f'Masking {len(np.where(active_data_mask))} of {np.size(active_data)} elements.')
        # apply the mask now:
        active_data[active_data_mask] = np.nan
    
    mesh = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    mesh["Elevation"] = (active_data.ravel(order="F") * zScalingFactor)

    plot_name = build_3d_plot_identifier_name('plot_3d_binned_bars', kwargs.get('name', ''))
    kwargs['name'] = plot_name # this is the only one to overwrite in kwargs
    # print(f'name: {plot_name}')    
    plotActor = p.add_mesh(mesh,
                            **({'show_edges': True, 'edge_color': 'k', 'nan_opacity': 0.0, 'scalars': 'Elevation', 'opacity': 1.0, 'use_transparency': False, 'smooth_shading': False, 'show_scalar_bar': False, 'render': True} | kwargs)
                          )
    # p.enable_depth_peeling() # this fixes bug where it appears transparent even when opacity is set to 1.00
    
    plotActors = {plot_name: {'main': plotActor}}
    data_dict = {plot_name: { 
            'name':plot_name,
            'grid': mesh, 
            'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
            'active_data': active_data
        }
    }
    return plotActors, data_dict
    


    
    
## Point Labeling:

def _perform_plot_point_labels(p, active_points, point_labels=None, point_mask=None, **kwargs):
    if point_mask is not None:
        if callable(point_mask):
            point_masking_function = point_mask
            point_mask = point_masking_function(active_points) # if it is a masking function instead of a list of inclusion bools, build the concrete list by evaluating it for active_points
            # point_mask = [point_masking_function(a_point) for a_point in active_points] 
        assert (len(point_mask) == len(active_points)), "There must be one mask value in point_mask for each point in active_points!"
        active_points = active_points[point_mask] # apply the mask to the points
    
    if point_labels is None:
        point_labels = [f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})' for a_point in active_points]
    if callable(point_labels):
        point_labeling_function = point_labels
        point_labels = [point_labeling_function(a_point) for a_point in active_points] # if it is a formatting function instead of a list of labels, build the concrete list by evaluating it over the active_points
    assert (len(point_labels) == len(active_points)), "There must be one point label in point_labels for each point in active_points!"
    
    points_labels_actor = p.add_point_labels(active_points, point_labels,
                                                **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)
                                             )
    plotActors = {'main': points_labels_actor}
    data_dict = {
        'point_labels': point_labels,
        'points': active_points
    }
    return plotActors, data_dict 
    
    
def plot_point_labels(p, xbin_centers, ybin_centers, data, point_labels=None, point_mask=None, zScalingFactor=1.0, **kwargs):
    """ Plots 3D text point labels at the provided points.

    Args:
        p ([type]): [description]
        xbin_centers ([type]): [description]
        ybin_centers ([type]): [description]
        data ([type]): [description]
        point_labels [str]: a set of labels of length equal to data to display on the points
        zScalingFactor (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
        
    Examples:
    # The full point shown:
    point_labeling_function = lambda (a_point): return f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})'
    # Only the z-values
    point_labeling_function = lambda a_point: f'{a_point[2]:.2f}'
    point_masking_function = lambda points: points[:, 2] > 20.0
    plotActors_CenterLabels, data_dict_CenterLabels = plot_point_labels(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers, active_epoch_placefields2D.ratemap.occupancy, 
                                                                        point_labels=point_labeling_function, 
                                                                        point_mask=point_masking_function,
                                                                        shape='rounded_rect', shape_opacity= 0.2, show_points=False)

    """
    # build a structured grid out of the bins
    twoDimGrid_x, twoDimGrid_y = np.meshgrid(xbin_centers, ybin_centers)
    active_data = data[:,:].T.copy() # A single tuning curve
    grid = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    points = grid.points
    
    plot_name = build_3d_plot_identifier_name('plot_point_labels', kwargs.get('name', 'main'))
    kwargs['name'] = plot_name # this is the only one to overwrite in kwargs
    plotActors_labels, data_dict_labels = _perform_plot_point_labels(p, points, point_labels=point_labels, point_mask=point_mask,
                                                                        **({'point_size': 8, 'font_size': 10, 'name': 'build_center_labels_test', 'shape_opacity': 0.8, 'show_points': False} | kwargs)
                                                                    )
    # plotActors = {'main': plotActors_labels['main']}
    
    plotActors = {plot_name: plotActors_labels['main']}
    data_dict = {plot_name: { 
            'name':plot_name,
            'grid': grid, 
            'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
            'active_data': active_data
        } | data_dict_labels
    }
    
    return plotActors, data_dict


