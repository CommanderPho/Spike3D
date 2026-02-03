import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Arrow



def example_quiver_2D(canvas, view):
	# Create a grid of arrow starting points (like meshgrid)
	x = np.linspace(-5, 5, 10)
	y = np.linspace(-5, 5, 10)
	X, Y = np.meshgrid(x, y)

	# Flatten the grid
	x_pos = X.flatten()
	y_pos = Y.flatten()

	# Define vector field (u, v components)
	u = -Y.flatten()  # x-direction
	v = X.flatten()   # y-direction

	# Scale arrows for visibility
	scale = 0.3

	# Build arrow matrix: each row is [x1, y1, x2, y2]
	# where (x1, y1) is start and (x2, y2) is end point
	arrows = np.column_stack([
		x_pos, y_pos,
		x_pos + u * scale, y_pos + v * scale
	])

	# Create arrow visual
	arrow_visual = Arrow(arrows=arrows, arrow_type='stealth', 
						arrow_size=10, color='red',
						arrow_color='red', width=2)
	view.add(arrow_visual)
	
	data_out = dict(arrow_visual=arrow_visual, arrows=arrows)
	
	return canvas, view, data_out


def example_quiver_3D(canvas, view):
	# Create 3D grid
	x = np.linspace(-2, 2, 5)
	y = np.linspace(-2, 2, 5)
	z = np.linspace(-2, 2, 5)
	X, Y, Z = np.meshgrid(x, y, z)

	# Flatten
	x_pos = X.flatten()
	y_pos = Y.flatten()
	z_pos = Z.flatten()

	# Define 3D vector field
	u = -y_pos
	v = x_pos
	w = np.zeros_like(z_pos)

	# Normalize and scale
	scale = 0.3

	# Build 3D arrow matrix
	arrows_3d = np.column_stack([
		x_pos, y_pos, z_pos,
		x_pos + u * scale, y_pos + v * scale, z_pos + w * scale
	])

	# Create 3D arrow visual
	view.camera = 'turntable'
	arrow_visual = Arrow(arrows=arrows_3d, arrow_type='triangle_60',
						arrow_size=5, color='blue')
	view.add(arrow_visual)
	data_out = dict(arrow_visual=arrow_visual, arrows=arrows_3d)
	return canvas, view, data_out



if __name__ == '__main__':
	# Create canvas
	canvas = scene.SceneCanvas(keys='interactive', show=True)
	view = canvas.central_widget.add_view()
	view.camera = 'panzoom'
	# canvas, view, data_out = example_quiver_2D(canvas=canvas, view=view)
	canvas, view, data_out = example_quiver_3D(canvas=canvas, view=view)
	app.run()

