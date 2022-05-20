
## Potentially Interesting Examples:
python examples\basic\mousehighlight.py ### Enables highlighting mesh on click
basic\colorlines.py
python examples\basic\clustering.py ### Renders clusters and removes outliers automatically somehow
python examples\basic\flatarrow.py ### Renders curved arrows that are flat in a given plane

### Generated Silhouettes/Projects of a mesh to a specified plane.
python examples\basic\silhouette1.py

### Broken due to missing meshes:
python examples\basic\glyphs.py

#### Multi-state buttons (as would be used for toggling place cells)
python examples\basic\buttons.py



## Potentially Interesting - Advanced

Use the Moving Least Squares algorithm to project a cloud of points to a smooth line
least_squares1d

Use the Moving Least Squares algorithm to project a cloud of points to a smooth surface
least_squares2d

Reconstruct a triangular mesh from a noisy cloud of points.
recosurface


Fit a line and plane to a 3D point cloud
python examples\advanced\fitline.py


Interpolate the arrays of a source mesh (RandomHills) onto another (ellipsoid) by averaging closest point values
interpolateMesh
python examples\advanced\interpolateMeshArray.py


A proper multi-window viewer:
python examples\advanced\multi_viewer1.py

Same as 1 but with lots of options:
python examples\advanced\multi_viewer2.py

A pyQt-based Window:
python examples\other\qt_window1.py


A minimal example of how to embed a rendering window into a Qt application. Includes interactive slider.
python examples\other\qt_window2.py

## Potentially Interesting - Volumetric

raycaster


Probe a volumetric dataset with a line and plot the intensity values
probeLine2

Create an image from a numpy array containing an alpha channel for opacity
python examples\volumetric\image_rgba.py


Build a volume from a mesh where the inside voxels are set to 1 and the outside voxels are set to 0
mesh2volume


## Simulations

Add a trailing line to a moving object
`python examples\simulations\trail.py`
Relevant to adding the recent trajectory path to the rat's position data


## ** MOST RELEVANT ** - Plotting
https://vedo.embl.es/#gallery

### 2D Plots:
python examples\pyplot\whiskers.py

python examples\pyplot\histo_violin.py

A 1D histogram of a single variable
python examples\pyplot\histo_1D.py

A 2.5D histogram of two variables (one on each axis: x & y). Displayed as a heatmap.
python examples\pyplot\histo_2D.py

A true 3D histogram (histogram of three independent variables). The size of the cube at each point is proportional to the value at that point.
python examples\pyplot\histo_3D.py

A 2.5D histogram of two variables (one on each axis: x & y) **with hexagonal bins**.
python examples\pyplot\histo_hexagonal.py

histo_gauss


# density2d - Density plot (placefield) from a distribution of points in 2D
python examples\pyplot\plot_density2d.py

# plot_density3d - Same thing but for distribution of points in 3D
python examples\pyplot\plot_density3d.py

# plot_density4d - The 3d points over time. Could be used to display time-varying placefields. Probably too complex though.
python examples\pyplot\plot_density4d.py



hexagonal
hexagonal

### 3D Plotting Plots:

#### Display glyphs (meshes) - useful for rendering 3D spikes on a spatial graph
python examples\pyplot\glyphs3.py

#### Time-dependent animated waveforms/lines - looks like an extracellular voltage measurement
python examples\pyplot\anim_lines.py

#### plot_spheric
python examples\pyplot\plot_spheric.py

#### Spherical Histogram - Surface plotting in spherical coordinates. Spherical harmonic function is Y(l=2, m=0)
python examples\pyplot\histo_spheric.py



## Kinda irrelevant but still cool:
python examples\simulations\multiple_pendulum.py
pendulum_ode

### A 3D Equation Plot:
python examples\simulations\volterra.py


# Observed Errors:
most seem to stem from not having the mesh.

[vedo.io.py:149] ERROR: in load(), cannot load http://vedo.embl.es/examples/data/images/schrod.png
Traceback (most recent call last):
  File "C:\Users\pho\repos\vedo\examples\simulations\tunnelling1.py", line 48, in <module>
    bck = plt.load(dataurl+"images/schrod.png").scale(0.015).pos([0, 0, -0.5])
AttributeError: 'NoneType' object has no attribute 'scale'

momentum (hence undefined position) in a box hitting a potential barrier.
[vedo.io.py:149] ERROR: in load(), cannot load http://vedo.embl.es/examples/data/images/schrod.png
Traceback (most recent call last):
  File "C:\Users\pho\repos\vedo\examples\simulations\tunnelling2.py", line 46, in <module>
    bck = plt.load(dataurl+"images/schrod.png").alpha(.3).scale(.0256).pos([0,-5,-.1])
AttributeError: 'NoneType' object has no attribute 'alpha'


## TypeError: __init__() got an unexpected keyword argument 's'
Appears to be a non-load(...) related error.

(phoviz_ultimate) C:\Users\pho\repos\vedo>python examples\simulations\doubleslit.py
Traceback (most recent call last):
  File "C:\Users\pho\repos\vedo\examples\simulations\doubleslit.py", line 26, in <module>
    screen = Grid(pos=[0, 0, -D], s=[0.1,0.1], lw=0, res=[200,50]).wireframe(False)
TypeError: __init__() got an unexpected keyword argument 's'

(phoviz_ultimate) C:\Users\pho\repos\vedo>python examples\simulations\drag_chain.py
Traceback (most recent call last):
  File "C:\Users\pho\repos\vedo\examples\simulations\drag_chain.py", line 22, in <module>
    surf = Plane(s=[60, 60])
TypeError: __init__() got an unexpected keyword argument 's'



## TypeError: __init__() got an unexpected keyword argument 'res'
(phoviz_ultimate) C:\Users\pho\repos\vedo>python examples\simulations\optics_main3.py
Traceback (most recent call last):
  File "C:\Users\pho\repos\vedo\examples\simulations\optics_main3.py", line 7, in <module>
    grid = Grid(res=[3,4])  # pick a few points in space to place cylinders
TypeError: __init__() got an unexpected keyword argument 'res'