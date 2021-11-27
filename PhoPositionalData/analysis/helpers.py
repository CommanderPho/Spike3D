import numpy as np


## Centroid point for camera
def centeroidnp(arr):
    # Calculate the centroid of an array of points
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


def min_max_bounds(arr):
    # Calculate the min and max of an array of points
    max_x = np.amax(arr[:, 0])
    max_y = np.amax(arr[:, 1])
    min_x = np.amin(arr[:, 0])
    min_y = np.amin(arr[:, 1])
    return [min_x, max_x, min_y, max_y]


def bounds_midpoint(arr):
    # calculates the (x, y) midpoint given input in the format [min_x, max_x, min_y, max_y]
    min_x, max_x, min_y, max_y = arr
    return [(min_x + max_x)/2.0, (min_y + max_y)/2.0]
