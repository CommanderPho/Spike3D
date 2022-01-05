import numpy as np
import pandas as pd

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


## Pandas DataFrame helpers:
def partition(df: pd.DataFrame, partitionColumn: str):
    # splits a DataFrame df on the unique values of a specified column (partitionColumn) to return a unique DataFrame for each unique value in the column.
    unique_values = np.unique(df[partitionColumn]) # array([ 0,  1,  2,  3,  4,  7, 11, 12, 13, 14])
    grouped_df = df.groupby([partitionColumn]) #  Groups on the specified column.
    return unique_values, np.array([grouped_df.get_group(aValue) for aValue in unique_values], dtype=object) # dataframes split for each unique value in the column


