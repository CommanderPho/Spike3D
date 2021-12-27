import numpy as np
import pyvista as pv


def prepare_binned_data_for_3d_bars(xbin, ybin, data):
    """ Sequentally repeats xbin, ybin, and data entries to prepare for being plot in 3D bar-plot form.
    Does this by repeating the xbin and ybin except the first and last entries so that there is one entry for each vertex of a 2d rectangular polygon.
    Following that, it repeats in both dimensions the data values, so that each of the created verticies has the same height value.
    Usage:
        modified_xbin, modified_ybin, modified_data = prepare_binned_data_for_3d(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
    """
    # duplicates every xbin value except for the first and last:
    modified_xbin = np.repeat(xbin, 2) # remove the first and last elements, which are duplicates
    modified_xbin = modified_xbin[1:-1] 
    modified_ybin = np.repeat(ybin, 2) # remove the first and last elements, which are duplicates
    modified_ybin = modified_ybin[1:-1]
    # there should be precisely double the number of bins in each direction as there are data
    print(f'np.shape(data): {np.shape(data)}, np.shape(modified_xbin): {np.shape(modified_xbin)}, np.shape(modified_ybin): {np.shape(modified_ybin)}')
    assert (np.shape(data)[0] * 2) == np.shape(modified_xbin)[0], "There shoud be double the number of xbins in the modified array as there are data points in the array."
    assert (np.shape(data)[1] * 2) == np.shape(modified_ybin)[0], "There shoud be double the number of ybins in the modified array as there are data points in the array."
    
    modified_data = np.repeat(data, 2, axis=0)
    modified_data = np.repeat(modified_data, 2, axis=1)
    return modified_xbin, modified_ybin, modified_data
    
    
    
def plot_3d_binned_data(plotter, xbin, ybin, data, zScalingFactor=1.0, drop_below_threshold: float=None):
    """ Plots a 3D bar-graph
    Usage:
        modified_xbin, modified_ybin, modified_data = prepare_binned_data_for_3d(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy)
        plot_3d_binned_data(pActiveTuningCurvesPlotter,modified_xbin, modified_ybin, modified_data)
    """
    # build a structured grid out of the bins
    twoDimGrid_x, twoDimGrid_y = np.meshgrid(xbin, ybin)
    # grid = pv.StructuredGrid(twoDimGrid)
    # grid = pv.RectilinearGrid(twoDimGrid)
    active_data = data[:,:].T.copy() # A single tuning curve
    mesh = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    mesh["Elevation"] = (active_data.ravel(order="F") * zScalingFactor)
    
    curr_opacity = None
    curr_smooth_shading = False
    plotActor = plotter.add_mesh(mesh, name='test', show_edges=True, edge_color='k', nan_opacity=0.0, scalars='Elevation', opacity=1.00, use_transparency=False, smooth_shading=False, show_scalar_bar=False, render=True)
    plotter.enable_depth_peeling() # this fixes bug where it appears transparent even when opacity is set to 1.00
    
    plotActors = {'main': plotActor}
    data_dict = {
        'mesh':mesh, 
        'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
        'active_data': active_data
    }
    return plotActors, data_dict
    
    

def plot_point_labels(p, active_points, point_labels=None):
    if point_labels is None:
        point_labels = [f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})' for a_point in active_points]
    assert (len(point_labels) == len(active_points)), "There must be one point label in point_labels for each point in active_points!"
    points_labels_actor = p.add_point_labels(active_points, point_labels, point_size=8, font_size=10, name='build_center_labels_test', shape_opacity=0.8, show_points=False)
    plotActors = {'main': points_labels_actor}
    data_dict = {
        'point_labels': point_labels
    }
    return plotActors, data_dict 
    
def build_center_labels(p, xbin_centers, ybin_centers, data, zScalingFactor=1.0):
    """[summary]

    Args:
        p ([type]): [description]
        xbin_centers ([type]): [description]
        ybin_centers ([type]): [description]
        data ([type]): [description]
        zScalingFactor (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    
    # build a structured grid out of the bins
    twoDimGrid_x, twoDimGrid_y = np.meshgrid(xbin_centers, ybin_centers)
    active_data = data[:,:].T.copy() # A single tuning curve
    grid = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    grid["Elevation"] = (active_data.ravel(order="F") * zScalingFactor)
    # points_actor = p.add_points(grid, name='build_center_labels_points', show_scalar_bar=False, render=True)
    points_actor = None
    # # mesh = pv.RectilinearGrid(twoDimGrid_x, twoDimGrid_y, active_data)
    # points = grid.points
    # mask = points[:, 0] > 0.1
    # if point_mask is not None:
    #     active_points = points[mask]
    # else:
    #     active_points = points.copy()
    # plot_point_labels(p, points, point_mask=mask)
    
    # mesh["Labels"] = (active_data.ravel(order="F") * zScalingFactor)
    # point_labels = [f'({a_point[0]:.2f}, {a_point[1]:.2f}, {a_point[2]:.2f})' for a_point in active_points]
    
    # Add string labels to the point data - this associates a label with every node:
    # mesh["Labels"] = [f"Label {i}" for i in range(mesh.n_points)]
    
    # Add the points with scalar labels:
    # p.add_point_scalar_labels(mesh, "Labels", name='build_center_labels_test', point_size=20, font_size=36)
    
    points = grid.points
    plotActors_labels, data_dict_labels = plot_point_labels(p, points)
    
    plotActors = {'main': points_actor, 'labels': plotActors_labels['main']}
    data_dict = {
        'grid': grid, 
        'twoDimGrid_x':twoDimGrid_x, 'twoDimGrid_y':twoDimGrid_y, 
        'active_data': active_data
    } | data_dict_labels
    return plotActors, data_dict


