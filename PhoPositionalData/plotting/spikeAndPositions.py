#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import pyvista as pv
import pyvistaqt as pvqt
import numpy as np
import pandas as pd
from pathlib import Path


# Fixed Geometry objects:
animal_location_sphere = pv.Sphere(radius=2.3)
animal_location_direction_cone = pv.Cone()
animal_location_circle = pv.Circle(radius=8.0)
animal_location_trail_circle = pv.Circle(radius=2.3)

## Spike indicator geometry:
spike_geom_cone = pv.Cone(direction=(0.0, 0.0, -1.0), height=10.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
# spike_geom_cone = pv.Cone(direction=(0.0, 0.0, 1.0), height=15.0, radius=0.2) # The spike geometry that is only displayed for a short while after the spike occurs
spike_geom_circle = pv.Circle(radius=0.4)
spike_geom_box = pv.Box(bounds=[-0.2, 0.2, -0.2, 0.2, -0.05, 0.05])
# pv.Cylinder

# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonCore import vtkLookupTable # required for build_custom_placefield_maps_lookup_table(...)



def build_repeated_spikes_color_array(spikes_df):
    
    spike_color_info.render_opacity
    flat_spike_colors_array = np.array([pv.parse_color(spike_color_info.rgb_hex, opacity=spike_color_info.render_opacity) for spike_color_info in spikes_df[['rgb_hex', 'render_opacity']].itertuples()])
    return flat_spike_colors_array
    
    
def build_custom_placefield_maps_lookup_table(curr_active_neuron_color, num_opacity_tiers, opacity_tier_values):
    """
    Usage:
        build_custom_placefield_maps_lookup_table(curr_active_neuron_color, 3, [0.0, 0.6, 1.0])
    """
    # opacity_tier_values: [0.0, 0.6, 1.0]
    # Build a simple lookup table of the curr_active_neuron_color with varying opacities
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(num_opacity_tiers)
    for i in np.arange(num_opacity_tiers):
        map_curr_active_neuron_color = curr_active_neuron_color.copy()
        map_curr_active_neuron_color[3] = opacity_tier_values[i]
        # print('map_curr_active_neuron_color: {}'.format(map_curr_active_neuron_color))
        lut.SetTableValue(i, map_curr_active_neuron_color)
    return lut




def _build_flat_arena_data(x, y, z=-0.01, smoothing=True, extrude_height=-5):
        # Builds the flat base maze map that the other data will be plot on top of
        ## Implicitly relies on: x, y
        # z = np.zeros_like(x)
        z = np.full_like(x, z) # offset just slightly in the z direction to account for the thickness of the caps that are added upon extrude
        point_cloud = np.vstack((x, y, z)).T
        pdata = pv.PolyData(point_cloud)
        pdata['occupancy heatmap'] = np.arange(np.shape(point_cloud)[0])
        # geo = pv.Circle(radius=0.5)
        # pc = pdata.glyph(scale=False, geom=geo)
        if smoothing:
            surf = pdata.delaunay_2d()
            surf = surf.extrude([0,0,extrude_height], capping=True, inplace=True)
            clipped_surf = surf.clip('-z', invert=False)
            return pdata, clipped_surf
        else:
            geo = pv.Circle(radius=0.5)
            pc = pdata.glyph(scale=False, geom=geo)
            return pdata, pc
        

def perform_plot_flat_arena(p, *args, z=-0.01, bShowSequenceTraversalGradient=False, smoothing=True, extrude_height=-5, **kwargs):
    """ Upgraded to render a much better looking 3D extruded maze surface. """
    # Call with:
    # pdata_maze, pc_maze = build_flat_map_plot_data() # Plot the flat arena
    # p.add_mesh(pc_maze, name='maze_bg', color="black", render=False)

    if len(args) == 2:
        # normal x, y case
        x, y = args[0], args[1]
        pdata_maze, pc_maze = _build_flat_arena_data(x, y, z=z, smoothing=smoothing, extrude_height=extrude_height)

    elif len(args) == 1:
        # directly passing in pc_maze already built by calling _build_flat_arena_data case
        # Note that  z, smoothing=smoothing, extrude_height=extrude_height are ignored in this case
        pc_maze = args[0]
    else:
        raise ValueError

    # return p.add_mesh(pc_maze, name='maze_bg', label='maze', color="black", show_edges=False, render=True)
    return p.add_mesh(pc_maze, **({'name': 'maze_bg', 'label': 'maze', 'color': [0.1, 0.1, 0.1], 'pbr': True, 'metallic': 0.8, 'roughness': 0.5, 'diffuse': 1, 'render': True} | kwargs))
    # return p.add_mesh(pc_maze, **({'name': 'maze_bg', 'label': 'maze', 'color': [0.1, 0.1, 0.1, 1.0], 'pbr': True, 'metallic': 0.8, 'roughness': 0.5, 'diffuse': 1, 'render': True} | kwargs))
    # bShowSequenceTraversalGradient
    if bShowSequenceTraversalGradient:
        traversal_order_scalars = np.arange(len(x))
        return p.add_mesh(pc_maze, **({'name': 'maze_bg', 'label': 'maze', 'scalars': traversal_order_scalars, 'render': True} | kwargs))



# dataframe version of the build_active_spikes_plot_pointdata(...) function
def build_active_spikes_plot_pointdata_df(active_flat_df: pd.DataFrame):
    """Builds the pv.PolyData pointcloud from the spikes dataframe points.

    Args:
        active_flat_df (pd.DataFrame): [description]

    Returns:
        [type]: [description]
    """
    if 'z' in active_flat_df.columns:
        # use custom override z-values
        print('build_active_spikes_plot_pointdata_df(...): Found custom z column! Using Data!!')
        assert np.shape(active_flat_df['z']) == np.shape(active_flat_df['x']), "custom z values must be the same shape as the x column"
        spike_history_point_cloud = active_flat_df[['x','y','z']].to_numpy()
    else:
        # no provided custom z value
        active_flat_df['z_fixed'] = np.full_like(active_flat_df['x'].values, 1.1) # Offset a little bit in the z-direction so we can see it
        spike_history_point_cloud = active_flat_df[['x','y','z_fixed']].to_numpy()
        
    ## Old way:
    # spike_series_positions = active_flattened_spike_positions_list
    # z_fixed = np.full_like(spike_series_positions[0,:], 1.1) # Offset a little bit in the z-direction so we can see it
    # spike_history_point_cloud = np.vstack((spike_series_positions[0,:], spike_series_positions[1,:], z_fixed)).T
    spike_history_pdata = pv.PolyData(spike_history_point_cloud)
    # spike_history_pdata['times'] = spike_series_times
    # spike_history_pdata['cellID'] = active_flat_df['unit_id'].values
    # spike_history_pdata['cellID'] = active_flat_df['unit_id'].values
    spike_history_pdata['cellID'] = active_flat_df['aclu'].values
    
    if 'render_opacity' in active_flat_df.columns:
        spike_history_pdata['render_opacity'] = active_flat_df['render_opacity'].values
        # spike_history_pdata['render_opacity'] = np.expand_dims(active_flat_df['render_opacity'].values, axis=1)
        # alternative might be repeating 4 times along the second dimension for no reason.
    else:
        print('no custom render_opacity set on dataframe.')
        
    # rebuild the RGB data from the dataframe:
    if (np.isin(['R','G','B','render_opacity'], active_flat_df.columns).all()):
        # RGB Only:
        # spike_history_pdata['rgb'] = active_flat_df[['R','G','B']].to_numpy()
        # TODO: could easily add the spike_history_pdata['render_opacity'] here as RGBA if we wanted.
        # RGB+A:
        spike_history_pdata['rgb'] = active_flat_df[['R','G','B','render_opacity']].to_numpy()
        print('successfully set custom rgb key from separate R, G, B columns in dataframe.')
    else:
        print('WARNING: DATAFRAME LACKS RGB VALUES!')

    return spike_history_pdata


# dataframe versions of the build_active_spikes_plot_data(...) function
def build_active_spikes_plot_data_df(active_flat_df: pd.DataFrame, spike_geom):
    """ 
    Usage:
        spike_history_pdata, spike_history_pc = build_active_spikes_plot_data_df(active_flat_df, spike_geom)
    """
    spike_history_pdata = build_active_spikes_plot_pointdata_df(active_flat_df)
    spike_history_pc = spike_history_pdata.glyph(scale=False, geom=spike_geom.copy()) # create many glyphs from the point cloud
    return spike_history_pdata, spike_history_pc



## compatability with pre 2021-11-28 implementations
def build_active_spikes_plot_pointdata(active_flattened_spike_identities, active_flattened_spike_positions_list):
    # spike_series_times = active_flattened_spike_times # currently unused
    spike_series_identities = active_flattened_spike_identities # currently unused
    spike_series_positions = active_flattened_spike_positions_list
    # z = np.zeros_like(spike_series_positions[0,:])
    z_fixed = np.full_like(spike_series_positions[0,:], 1.1) # Offset a little bit in the z-direction so we can see it
    spike_history_point_cloud = np.vstack((spike_series_positions[0,:], spike_series_positions[1,:], z_fixed)).T
    spike_history_pdata = pv.PolyData(spike_history_point_cloud)
    # spike_history_pdata['times'] = spike_series_times
    spike_history_pdata['cellID'] = spike_series_identities
    return spike_history_pdata

## compatability with pre 2021-11-28 implementations
def build_active_spikes_plot_data(active_flattened_spike_identities, active_flattened_spike_positions_list, spike_geom):
    # spike_series_times = active_flattened_spike_times # currently unused
    spike_history_pdata = build_active_spikes_plot_pointdata(active_flattened_spike_identities, active_flattened_spike_positions_list)
    # create many spheres from the point cloud
    spike_history_pc = spike_history_pdata.glyph(scale=False, geom=spike_geom.copy())
    return spike_history_pdata, spike_history_pc



## This light effect occurs when a spike happens to indicate its presence
light_spawn_constant_z_offset = 2.5
light_spawn_constant_z_focal_position = -0.5 # by default, the light focuses under the floor

def build_spike_spawn_effect_light_actor(p, spike_position, spike_unit_color='white'):
    # spike_position: should be a tuple like (0, 0, 10)
    light_source_position = spike_position
    light_source_position[3] = light_source_position[3] + light_spawn_constant_z_offset
    light_focal_point = spike_position
    light_focal_point[3] = light_focal_point[3] + light_spawn_constant_z_focal_position
    
    SpikeSpawnEffectLight = pv.Light(position=light_source_position, focal_point=light_focal_point, color=spike_unit_color)
    SpikeSpawnEffectLight.positional = True
    SpikeSpawnEffectLight.cone_angle = 40
    SpikeSpawnEffectLight.exponent = 10
    SpikeSpawnEffectLight.intensity = 3
    SpikeSpawnEffectLight.show_actor()
    p.add_light(SpikeSpawnEffectLight)
    return SpikeSpawnEffectLight # return the light actor for removal later



def force_plot_ignore_scalar_as_color(plot_mesh_actor, lookup_table):
        """The following custom lookup table solution is required to successfuly plot the surfaces with opacity dependant on their scalars property and still have a consistent color (instead of using the scalars for the color too). Note that the previous "fix" for the problem of the scalars determining the object's color when I don't want them to:
        Args:
            plot_mesh_actor ([type]): [description]
            lookup_table ([type]): a lookup_table as might be built with: `build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.0, 0.6, 1.0])`
        """
        # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 5, [0.0, 0.0, 0.3, 0.5, 0.1])
        lookup_table.SetTableRange(plot_mesh_actor.GetMapper().GetScalarRange())
        lookup_table.Build()
        plot_mesh_actor.GetMapper().SetLookupTable(lookup_table)
        plot_mesh_actor.GetMapper().SetScalarModeToUsePointData()


def plot_placefields2D(pTuningCurves, active_placefields, pf_colors: np.ndarray, zScalingFactor=10.0, show_legend=False):
    """ Plots 2D Placefields in a 3D PyVista plot """
    # active_placefields: Pf2D    
    should_force_placefield_custom_color = True
    should_use_normalized_tuning_curves = True
    should_pdf_normalize_manually = False
    
    if should_use_normalized_tuning_curves:
        curr_tuning_curves = active_placefields.ratemap.normalized_tuning_curves.copy()
    else:
        curr_tuning_curves = active_placefields.ratemap.tuning_curves.copy()


    if np.shape(pf_colors)[1] > 3:
        opaque_pf_colors = pf_colors[0:3,:].copy() # get only the RGB values, discarding any potnential alpha information
    else:
        opaque_pf_colors = pf_colors.copy()
        
    # curr_tuning_curves[curr_tuning_curves < 0.1] = np.nan
    # curr_tuning_curves = curr_tuning_curves * zScalingFactor
    
    num_curr_tuning_curves = len(curr_tuning_curves)
    # Get the cell IDs that have a good place field mapping:
    good_placefield_neuronIDs = np.array(active_placefields.ratemap.neuron_ids) # in order of ascending ID
    tuningCurvePlot_x, tuningCurvePlot_y = np.meshgrid(active_placefields.ratemap.xbin_centers, active_placefields.ratemap.ybin_centers)
    # Loop through the tuning curves and plot them:
    print('num_curr_tuning_curves: {}'.format(num_curr_tuning_curves))
    tuningCurvePlotActors = []
    tuningCurvePlotData = []
    for i in np.arange(num_curr_tuning_curves):
        curr_active_neuron_ID = good_placefield_neuronIDs[i]
        curr_active_neuron_color = pf_colors[:, i]
        curr_active_neuron_opaque_color = opaque_pf_colors[:,i]
        curr_active_neuron_pf_identifier = 'pf[{}]'.format(curr_active_neuron_ID)
        curr_active_neuron_tuning_Curve = np.squeeze(curr_tuning_curves[i,:,:]).T.copy() # A single tuning curve
        
        if should_pdf_normalize_manually:
            # Normalize the area under the curve to 1.0 (like a probability density function)
            curr_active_neuron_tuning_Curve = curr_active_neuron_tuning_Curve / np.nansum(curr_active_neuron_tuning_Curve)
            
        curr_active_neuron_tuning_Curve = curr_active_neuron_tuning_Curve * zScalingFactor
        
        # curr_active_neuron_tuning_Curve[curr_active_neuron_tuning_Curve < 0.1] = np.nan
        pdata_currActiveNeuronTuningCurve = pv.StructuredGrid(tuningCurvePlot_x, tuningCurvePlot_y, curr_active_neuron_tuning_Curve)
        pdata_currActiveNeuronTuningCurve["Elevation"] = (curr_active_neuron_tuning_Curve.ravel(order="F") * zScalingFactor)
        
        curr_active_neuron_plot_data = {'curr_active_neuron_ID':curr_active_neuron_ID,
                                         'curr_active_neuron_pf_identifier':curr_active_neuron_pf_identifier,
                                         'curr_active_neuron_tuning_Curve':curr_active_neuron_tuning_Curve,'pdata_currActiveNeuronTuningCurve':pdata_currActiveNeuronTuningCurve}
        
        # contours_currActiveNeuronTuningCurve = pdata_currActiveNeuronTuningCurve.contour()
        # pdata_currActiveNeuronTuningCurve.plot(show_edges=True, show_grid=True, cpos='xy', scalars=curr_active_neuron_tuning_Curve.T)        
        # actor_currActiveNeuronTuningCurve = pTuningCurves.add_mesh(pdata_currActiveNeuronTuningCurve, label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier, show_edges=False, nan_opacity=0.0, color=curr_active_neuron_color, use_transparency=True)

        # surf = poly.delaunay_2d()
        # pTuningCurves.add_mesh(surf, label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier, show_edges=False, nan_opacity=0.0, color=curr_active_neuron_color, opacity=0.9, use_transparency=False, smooth_shading=True)
        if should_force_placefield_custom_color:
            curr_opacity = 'sigmoid'
            curr_smooth_shading = True
        else:
            curr_opacity = None
            curr_smooth_shading = False
            
        # curr_opacity = None
        # curr_smooth_shading = False
        
        # print(f'curr_active_neuron_color: {curr_active_neuron_color} for i: {i}')
        
        pdata_currActiveNeuronTuningCurve_plotActor = pTuningCurves.add_mesh(pdata_currActiveNeuronTuningCurve, label=curr_active_neuron_pf_identifier, name=curr_active_neuron_pf_identifier,
                                                                            show_edges=True, edge_color=curr_active_neuron_opaque_color, nan_opacity=0.0, scalars='Elevation', opacity=curr_opacity, use_transparency=False, smooth_shading=curr_smooth_shading, show_scalar_bar=False, render=False)                                                                     
        
        # Force custom colors:
        if should_force_placefield_custom_color:
            ## The following custom lookup table solution is required to successfuly plot the surfaces with opacity dependant on their scalars property and still have a consistent color (instead of using the scalars for the color too). Note that the previous "fix" for the problem of the scalars determining the object's color when I don't want them to:
                #   pdata_currActiveNeuronTuningCurve_plotActor.GetMapper().ScalarVisibilityOff() # Scalars not used to color objects
            # Is NOT Sufficient, as it disables any opacity at all seemingly
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 2, [0.2, 0.8])
            
            lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 1, [1.0])
            
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.2, 0.6, 1.0])
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 3, [0.0, 0.6, 1.0])
            # lut = build_custom_placefield_maps_lookup_table(curr_active_neuron_color.copy(), 5, [0.0, 0.0, 0.3, 0.5, 0.1])
            curr_active_neuron_plot_data['lut'] = lut
            force_plot_ignore_scalar_as_color(pdata_currActiveNeuronTuningCurve_plotActor, lut)
        
        # pTuningCurves.add_mesh(contours_currActiveNeuronTuningCurve, color=curr_active_neuron_color, line_width=1, name='{}_contours'.format(curr_active_neuron_pf_identifier))
        tuningCurvePlotActors.append(pdata_currActiveNeuronTuningCurve_plotActor)
        tuningCurvePlotData.append(curr_active_neuron_plot_data)
        
    # Legend:
    plots_data = {'good_placefield_neuronIDs': good_placefield_neuronIDs,
                'unit_labels': ['{}'.format(good_placefield_neuronIDs[i]) for i in np.arange(num_curr_tuning_curves)],
                 'legend_entries': [['pf[{}]'.format(good_placefield_neuronIDs[i]), opaque_pf_colors[:,i]] for i in np.arange(num_curr_tuning_curves)]}
    
    # lost the ability to have colors with alpha components
        # TypeError: SetEntry argument 4: expected a sequence of 3 values, got 4 values
        
    # lost the ability to specify exact origins in add_legend() # used to be origin=[0.95, 0.1]

    if show_legend:
        legendActor = pTuningCurves.add_legend(plots_data['legend_entries'], name='tuningCurvesLegend', 
                                bcolor=(0.05, 0.05, 0.05), border=True,
                                loc='center right', size=[0.05, 0.85]) # vtk.vtkLegendBoxActor
        
        # used to be origin=[0.95, 0.1]
        
    else:
        legendActor = None
        
    # pTuningCurves.show_grid()
    # pTuningCurves.add_axes(line_width=5, labels_off=False)
    # pTuningCurves.enable_depth_peeling(number_of_peels=num_curr_tuning_curves)
    # pTuningCurves.enable_3_lights()
    # pTuningCurves.enable_shadows()
    return pTuningCurves, tuningCurvePlotActors, tuningCurvePlotData, legendActor, plots_data

def update_plotVisiblePlacefields2D(tuningCurvePlotActors, isTuningCurveVisible):
    # Updates the visible placefields. Complements plot_placefields2D
    num_active_tuningCurveActors = len(tuningCurvePlotActors)
    for i in np.arange(num_active_tuningCurveActors):
        # tuningCurvePlotActors[i].SetVisibility(isTuningCurveVisible[i])
        if isTuningCurveVisible[i]:
            # tuningCurvePlotActors[i].show_actor()
            # tuningCurvePlotActors[i].SetVisibility(True)
            tuningCurvePlotActors[i].VisibilityOn()
        else:
            tuningCurvePlotActors[i].VisibilityOff()
            # tuningCurvePlotActors[i].hide_actor()
    
    