import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# import mplcursors
import math # For color map generation
from matplotlib.colors import ListedColormap
# from matplotlib.cm import get_cmap


# Neuropy
from neuropy.plotting.ratemaps import compute_data_extent, compute_data_aspect_ratio, corner_points_from_extents

# PhoPy3DPositionAnalysis2021:
from PhoPositionalData.plot_data import get_cmap



class ImagePlaneRendering:
    """ Implementor renders a 2D image or plot on a plane in 3D space using pyvista
    from PhoGui.InteractivePlotter.Mixins.ImagePlaneRendering import ImagePlaneRendering
    
    Standalone Usage:
    # Texture from file:
        image_file = r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'
        loaded_image_tex = pv.read_texture(image_file)

        pActiveImageTestPlotter = pvqt.BackgroundPlotter()
        ImagePlaneRendering.plot_3d_image(pActiveImageTestPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy, loaded_image_tex=loaded_image_tex)
    """
    
    @staticmethod
    def plot_3d_image(p, xpoints, ypoints, data, z_location=0.0, loaded_image_tex=None):
        """ Renders a 2D image such as a heatmap as a 3D plane
  
        Usage:
            # Texture from file:
            image_file = r'output\2006-6-07_11-26-53\maze\speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.10\pf2D-Occupancy-maze-odd_laps-speedThresh_0.00-gridBin_5.00_3.00-smooth_0.00_0.00-frateThresh_0.png'
            loaded_image_tex = pv.read_texture(image_file)
        
            plot_3d_image(pActiveTuningCurvesPlotter, active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.occupancy, loaded_image_tex=loaded_image_tex)
          """
        def _test_image(xx, yy):
            # create an image using numpy,
            # xx, yy = np.meshgrid(np.linspace(xmin, xmax, 2), np.linspace(ymin, ymax, 2))
            A, b = 500, 100
            zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

            # Creating a custom RGB image
            cmap = get_cmap(len(xx), name='nipy_spectral')
            norm = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
            hue = norm(zz.ravel())
            colors = (cmap(hue)[:, 0:3] * 255.0).astype(np.uint8)
            image = colors.reshape((xx.shape[0], xx.shape[1], 3), order="F")
            return image
        
        def _build_image(xx, yy, zz):
            # Creating a custom RGB image
            cmap = get_cmap(20, name='nipy_spectral')
            norm = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
            hue = norm(zz.ravel())
            colors = (cmap(hue)[:, 0:3] * 255.0).astype(np.uint8)
            image = colors.reshape((xx.shape[0], xx.shape[1], 3), order="F")
            return image

        extent2D = compute_data_extent(xpoints, ypoints)
        corner_points = corner_points_from_extents(extent2D)
        corner_points_x = corner_points[:,0]
        corner_points_y = corner_points[:,1]

        # Draw a plane with corners at the specified points:
        # build a structured grid out of the bins
        # twoDimGrid_x, twoDimGrid_y = np.meshgrid(xpoints, ypoints)
        twoDimGrid_x, twoDimGrid_y = np.meshgrid(corner_points_x, corner_points_y)
        # print(f'np.shape(twoDimGrid_x): {np.shape(twoDimGrid_x)}, np.shape(twoDimGrid_y): {np.shape(twoDimGrid_y)}')
            
        # grid = pv.StructuredGrid(twoDimGrid)
        # grid = pv.RectilinearGrid(twoDimGrid)
        # active_data = data[:,:].T.copy() # A single tuning curve
        # z_data = np.full_like(active_data, z_location)
        
        z_data = np.full_like(twoDimGrid_x, z_location)
        mesh = pv.StructuredGrid(twoDimGrid_x, twoDimGrid_y, z_data)
        # mesh["Elevation"] = (active_data.ravel(order="F") * zScalingFactor)
        # Map the curved surface to a plane - use best fitting plane
        mesh.texture_map_to_plane(inplace=True)

        # if an image texture is provided directly, use that, otherwise build one programmatically (TODO)
        if loaded_image_tex is not None:
            tex = loaded_image_tex
        else:
            xmin, xmax, ymin, ymax = extent2D
            image = _test_image(np.linspace(xmin, xmax, 20), np.linspace(ymin, ymax, 20))
            image = _test_image(twoDimGrid_x, twoDimGrid_y)
            image = _build_image(twoDimGrid_x, twoDimGrid_y, data)
            # Get the 2D image as a texture to apply to the plane:
            # Convert 3D numpy array to texture
            tex = pv.numpy_to_texture(image)

        plotActor = p.add_mesh(mesh, name='plot_3d_image_test', texture=tex, show_edges=True, edge_color='k', nan_opacity=0.0, scalars=None, opacity=1.00, use_transparency=False, smooth_shading=False, show_scalar_bar=False, render=True)
        p.enable_depth_peeling() # this fixes bug where it appears transparent even when opacity is set to 1.00
        return plotActor

