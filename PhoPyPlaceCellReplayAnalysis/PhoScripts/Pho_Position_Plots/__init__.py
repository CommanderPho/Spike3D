#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""

## Plot Function Definitions:
# %%
## Colored Line:
# https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
# http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
import numpy as np
from scipy.stats import kde
import matplotlib.pyplot as plt
import matplotlib as mpl  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.animation as animation

# For Gradient Line Plotting:
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import seaborn as sns



def plot_coloredLine(x, y):
    def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
        """
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """

        def make_segments(x, y):
            """
            Create list of line segments from x and y coordinates, in the correct format
            for LineCollection: an array of the form numlines x (points per line) x 2 (x
            and y) array
            """

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments

        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])

        z = np.asarray(z)

        segments = make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha)

        ax = plt.gca()
        ax.add_collection(lc)

        return lc

    # Begin main function
    fig, ax = plt.subplots()

    z = np.linspace(0, 1, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=1, alpha=0.75)

    plt.xlim(np.nanmin(x), np.nanmax(x))
    plt.ylim(np.nanmin(y), np.nanmax(y))
    # plt.ylim(-1.0, 1.0)

    plt.show()

# 3D Plot:
def plot_animated3DPlot(t, x, y):
    # %%
    ## Animated 3D Plot:
    # From https://www.bragitoff.com/2020/10/3d-trajectory-animated-using-matplotlib-python/

    # References
    # https://gist.github.com/neale/e32b1f16a43bfdc0608f45a504df5a84
    # https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
    # https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation

    # ANIMATION FUNCTION
    def func(num, dataSet, line):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(dataSet[0:2, :num])
        line.set_3d_properties(dataSet[2, :num])
        return line


    # THE DATA POINTS
    # t = np.arange(0,20,0.2) # This would be the z-axis ('t' means time here)
    # x = np.cos(t)-1
    # y = 1/2*(np.cos(2*t)-1)
    dataSet = np.array([x, y, t])
    numDataPoints = len(t)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    # ax = Axes3D(fig, auto_add_to_figure=False)
    ax = fig.add_subplot(111, projection='3d')
    fig.add_axes(ax)


    # NOTE: Can't pass empty arrays into 3d version of plot()
    line = plt.plot(dataSet[0], dataSet[1], dataSet[2], lw=2, c='g')[0] # For line plot

    # AXES PROPERTIES]
    # ax.set_xlim3d([limit0, limit1])
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('time')
    ax.set_title('Trajectory of rat')

    # Creating the Animation object
    # line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataSet,line), interval=50, blit=False)
    # line_ani.save(r'AnimationNew.mp4')

    plt.show()

# %%
# Plot Type 1
def plot_type_one(x, y):
    # Create a figure with 6 plot areas
    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))

    # Everything starts with a Scatterplot
    axes[0].set_title('Scatterplot')
    axes[0].plot(x, y, 'ko')
    # As you can see there is a lot of overlapping here!

    # Thus we can cut the plotting window in several hexbins
    nbins = 20
    axes[1].set_title('Hexbin')
    axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)

    # 2D Histogram
    axes[2].set_title('2D Histogram')
    axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    # plot a density
    axes[3].set_title('Calculate Gaussian KDE')
    axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap=plt.cm.BuGn_r)

    # add shading
    axes[4].set_title('2D Density with shading')
    axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)

    # contour
    axes[5].set_title('Contour')
    axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[5].contour(xi, yi, zi.reshape(xi.shape) )

    return fig

def plot_spikes_raster(data_timestamps, spike_ind, neuron_ind, start_time):
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap('tab20')

    # cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
    #                         'Dark2', 'Set1', 'Set2', 'Set3',
    #                         'tab10', 'tab20', 'tab20b', 'tab20c']

    # c = [cmap.colors[ind] for ind in neuron_ind] # This doesn't work because there's only 20 colors
    c = [cmap.colors[ind % 20] for ind in neuron_ind]
    ax.scatter(data_timestamps[spike_ind], neuron_ind + 1, c=c, s=5)
    # ax.scatter(data_timestamps[spike_ind], neuron_ind + 1, s=5)
    # ax.scatter(data_timestamps[spike_ind], neuron_ind + 1, s=5)

    ax.set_yticks((1, data_spikes.shape[1]))
    ax.set_ylim((1, data_spikes.shape[1]))
    ax.set_ylabel('Cells')

    ax.set_xlabel('Time [s]')

    # ax.set_xlim((0.0, 90.0))
    ax.set_xlim((start_time, start_time + 90.0))
    sns.despine(offset=5)
    return fig
