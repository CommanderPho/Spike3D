#!/usr/bin/env python
# coding: utf-8
"""
Example: Render a stack of 2D mask images onto a single 3D axis using Silx.

This script demonstrates how to:
1. Create or load a stack of 2D mask images
2. Stack them into a 3D volume
3. Render the volume using Silx's 3D scalar field viewer
4. Visualize masks with isosurfaces and cut planes
"""

import sys
import numpy
from pathlib import Path
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items
from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d import SFViewParamTree


def create_dummy_mask_stack(num_slices=20, height=256, width=256):
    """Create a dummy stack of 2D mask images for demonstration.
    
    :param int num_slices: Number of 2D slices in the stack
    :param int height: Height of each mask image
    :param int width: Width of each mask image
    :return: 3D numpy array of shape (num_slices, height, width)
    """
    stack = numpy.zeros((num_slices, height, width), dtype=numpy.float32)
    
    # Create circular masks at different z positions
    center_y, center_x = height // 2, width // 2
    for z in range(num_slices):
        # Create a circular mask with varying radius
        y, x = numpy.ogrid[:height, :width]
        radius = 30 + 20 * numpy.sin(z * numpy.pi / num_slices)
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        stack[z, mask] = 1.0
        
        # Add some variation - offset center for some slices
        if z % 3 == 0:
            offset_y = int(20 * numpy.cos(z * numpy.pi / 10))
            offset_x = int(20 * numpy.sin(z * numpy.pi / 10))
            y2, x2 = numpy.ogrid[:height, :width]
            mask2 = ((x2 - center_x - offset_x) ** 2 + (y2 - center_y - offset_y) ** 2) <= (radius * 0.6) ** 2
            stack[z, mask2] = 0.5
    
    return stack


def render_mask_stack_with_scene_window(mask_stack, use_isosurface=True):
    """Render mask stack using SceneWindow (more flexible, lower-level API).
    
    :param numpy.ndarray mask_stack: 3D array of shape (z, y, x) containing mask data
    :param bool use_isosurface: If True, render as isosurface; if False, use cut plane


	from Spike3D.LibrariesExamples.Silx.silx_examples.renderMaskStack3D import render_mask_stack_with_scene_window

    """
    qapp = qt.QApplication.instance() or qt.QApplication([])
    
    window = SceneWindow()
    sceneWidget = window.getSceneWidget()
    sceneWidget.setBackgroundColor((0.1, 0.1, 0.1, 1.0))
    sceneWidget.setForegroundColor((1.0, 1.0, 1.0, 1.0))
    sceneWidget.setTextColor((1.0, 1.0, 1.0, 1.0))
    
    # Create a 3D scalar field item and set its data
    # Note: Silx expects data in (z, y, x) format which matches our mask_stack
    volume = items.ScalarField3D()
    volume.setData(mask_stack)
    sceneWidget.addItem(volume)
    
    # Set volume transform to scale appropriately
    z_size, y_size, x_size = mask_stack.shape
    scale_x = 1.0
    scale_y = 1.0
    scale_z = 1.0  # Adjust z-scale if slices are spaced differently
    volume.setScale(scale_x, scale_y, scale_z)
    
    # Center the volume
    volume.setTranslation(-x_size/2, -y_size/2, -z_size/2)
    
    if use_isosurface:
        # Add isosurfaces at different threshold levels
        # For binary masks, use 0.5 as threshold
        volume.addIsosurface(0.5, '#FF000080')  # Red, semi-transparent
        volume.addIsosurface(0.75, '#00FF0080')  # Green, semi-transparent
    else:
        # Use cut plane for visualization
        cutPlane = volume.getCutPlanes()[0]
        cutPlane.setVisible(True)
        cutPlane.getColormap().setName('viridis')
        cutPlane.setNormal((0.0, 0.0, 1.0))  # Cut along z-axis
        cutPlane.moveToCenter()
    
    window.show()
    sys.excepthook = qt.exceptionHandler
    qapp.exec()


def render_mask_stack_with_scalar_field_view(mask_stack, isolevel=0.5):
    """Render mask stack using ScalarFieldView (higher-level, specialized for volumes).
    
    :param numpy.ndarray mask_stack: 3D array of shape (z, y, x) containing mask data
    :param float isolevel: Iso-surface level for rendering (default 0.5 for binary masks)
    """
    qapp = qt.QApplication.instance() or qt.QApplication([])
    
    # Create the viewer main window
    window = ScalarFieldView()
    
    # Create a parameter tree for the scalar field view
    treeView = SFViewParamTree.TreeView(window)
    treeView.setSfView(window)
    
    # Add the parameter tree to the main window in a dock widget
    dock = qt.QDockWidget()
    dock.setWindowTitle('Parameters')
    dock.setWidget(treeView)
    window.addDockWidget(qt.Qt.RightDockWidgetArea, dock)
    
    # Set ScalarFieldView data
    window.setData(mask_stack)
    
    # Set scale of the data (adjust if needed for aspect ratio)
    window.setScale(1.0, 1.0, 1.0)
    
    # Set axes labels
    window.setAxesLabels('X', 'Y', 'Z')
    
    # Add an iso-surface at the mask threshold
    window.addIsosurface(isolevel, '#FF0000FF')  # Red, fully opaque
    
    window.show()
    sys.excepthook = qt.exceptionHandler
    qapp.exec()


def load_mask_stack_from_list(mask_images):
    """Load a stack of 2D mask images from a list of numpy arrays.
    
    :param list mask_images: List of 2D numpy arrays, each representing a mask image
    :return: 3D numpy array of shape (len(mask_images), height, width)
    """
    if not mask_images:
        raise ValueError("mask_images list is empty")
    
    # Ensure all masks have the same shape
    first_shape = mask_images[0].shape
    for i, mask in enumerate(mask_images):
        if mask.shape != first_shape:
            raise ValueError(f"Mask {i} has shape {mask.shape}, expected {first_shape}")
    
    # Stack masks into 3D array
    stack = numpy.stack(mask_images, axis=0)
    return stack.astype(numpy.float32)


def main():
    """Main function demonstrating mask stack rendering."""
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Render a stack of 2D mask images in 3D using Silx")
    parser.add_argument('--viewer', choices=['scene', 'scalar'], default='scene',
                        help='Type of viewer to use: scene (SceneWindow) or scalar (ScalarFieldView)')
    parser.add_argument('--slices', type=int, default=20,
                        help='Number of slices in the dummy mask stack (default: 20)')
    parser.add_argument('--height', type=int, default=256,
                        help='Height of each mask image (default: 256)')
    parser.add_argument('--width', type=int, default=256,
                        help='Width of each mask image (default: 256)')
    parser.add_argument('--isolevel', type=float, default=0.5,
                        help='Iso-surface level for rendering (default: 0.5)')
    parser.add_argument('--no-isosurface', action='store_true',
                        help='Use cut plane instead of isosurface (only for scene viewer)')
    
    args = parser.parse_args()
    
    # Create dummy mask stack
    # print(f"Creating dummy mask stack: {args.slices} slices, {args.height}x{args.width}")
    # mask_stack = create_dummy_mask_stack(num_slices=args.slices, height=args.height, width=args.width)


    example_data_path = Path('LibrariesExamples/Silx/example_data').resolve()
    example_data_file = example_data_path.joinpath(f'example_t_bin_masks.npy').resolve()
    assert example_data_file.exists()
    print(f"Loading mask from file: {example_data_file}...")
    mask_stack = np.load(example_data_file)
    print(f'\tdone.\n\tmask_stack: {np.shape(mask_stack)}')


    print(f"Mask stack shape: {mask_stack.shape}")
    print(f"Mask stack range: [{mask_stack.min():.2f}, {mask_stack.max():.2f}]")
    
    # Render based on viewer type
    if args.viewer == 'scene':
        render_mask_stack_with_scene_window(mask_stack, use_isosurface=not args.no_isosurface)
    else:
        render_mask_stack_with_scalar_field_view(mask_stack, isolevel=args.isolevel)


if __name__ == '__main__':
    main()
