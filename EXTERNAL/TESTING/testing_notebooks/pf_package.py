"""Standalone package for plotting placefields"""

from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt


def load_pf(pf_file):

    with open(pf_file, 'rb') as file:
        PF = load(file)

    if type(PF.sr_image) is not int:
        PF.sr_image = PF.sr_image.squeeze()  # Backwards compatibility fix

    return PF


def plot_tmap_us(obj, ax_ind):
    """
    Plot unsmoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_us[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_neurons - 1
    obj.ax[ax_ind].axis('off')


def plot_tmap_sm(obj, ax_ind):
    """
    Plot smoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_sm[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_neurons - 1
    obj.ax[ax_ind].axis('off')
    obj.ax[ax_ind].set_title(obj.titles[obj.current_position])


def plot_events_over_pos(obj, ax_ind):
    """
    Plot trajectory with calcium events

    """
    psa_use = obj.PSAbool[obj.current_position, :]
    obj.ax[ax_ind].plot(obj.x, obj.y, 'k-')
    obj.ax[ax_ind].plot(obj.x[psa_use == 1], obj.y[psa_use == 1], 'r*')
    obj.ax[ax_ind].set_xlim(obj.traj_lims[0])
    obj.ax[ax_ind].set_ylim(obj.traj_lims[1])
    obj.last_position = obj.n_neurons - 1
    obj.ax[ax_ind].set_title(obj.mouse + ' ' + obj.arena + ' Day ' + str(obj.day))
    obj.ax[ax_ind].axis('off')
    # print("sum psa_use = " + str(np.sum(psa_use)))


def plot_psax(obj, ax_ind):
    """
    Plot trajectory in x-direction with calcium events

    """
    psa_use = obj.PSAbool[obj.current_position, :]
    time_use = np.asarray((np.arange(len(psa_use)) + 1)/obj.sample_rate)
    obj.ax[ax_ind].plot(time_use, obj.x, 'k-')
    obj.ax[ax_ind].plot(time_use[psa_use == 1], obj.x[psa_use == 1], 'r*')
    obj.ax[ax_ind].set_xlabel('Time (s)')
    obj.ax[ax_ind].set_ylabel('X position (cm)')


def plot_psay(obj, ax_ind):
    """
    Plot trajectory in y-direction with calcium events

    """
    psa_use = obj.PSAbool[obj.current_position, :]
    time_use = np.asarray((np.arange(len(psa_use)) + 1)/obj.sample_rate)
    obj.ax[ax_ind].plot(time_use, obj.y, 'k-')
    obj.ax[ax_ind].plot(time_use[psa_use == 1], obj.y[psa_use == 1], 'r*')
    obj.ax[ax_ind].set_xlabel('Time (s)')
    obj.ax[ax_ind].set_ylabel('Y position (cm)')


def plot_tmap_us2(obj, ax_ind):
    """
    Plot 2nd unsmoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_us2[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_neurons - 1
    obj.ax[ax_ind].set_title('rho_spear = ' + str(round(obj.corrs_us[obj.current_position], 3)))
    obj.ax[ax_ind].axis('off')


def plot_tmap_sm2(obj, ax_ind):
    """
    Plot 2nd smoothed tmap
    :param obj:
    :return:
    """

    obj.ax[ax_ind].imshow(obj.tmap_sm2[obj.current_position], cmap='viridis')
    obj.last_position = obj.n_neurons - 1
    obj.ax[ax_ind].axis('off')
    obj.ax[ax_ind].set_title('rho_spear = ' + str(round(obj.corrs_sm[obj.current_position], 3)))


def plot_events_over_pos2(obj, ax_ind):
    """
    Plot 2nd trajectory with calcium events

    """
    psa_use = obj.PSAbool2[obj.current_position, :]
    obj.ax[ax_ind].plot(obj.x2, obj.y2, 'k-')
    obj.ax[ax_ind].plot(obj.x2[psa_use == 1], obj.y2[psa_use == 1], 'r*')
    obj.ax[ax_ind].set_xlim(obj.traj_lims2[0])
    obj.ax[ax_ind].set_ylim(obj.traj_lims2[1])
    obj.last_position = obj.n_neurons - 1
    obj.ax[ax_ind].set_title(obj.arena2 + ' Day ' + str(obj.day2))
    obj.ax[ax_ind].axis('off')
    # print("sum psa_use = " + str(np.sum(psa_use)))



class PlaceFieldObject:
    def __init__(self, tmap_us, tmap_gauss, xrun, yrun, PSAboolrun, occmap, runoccmap,
                 xEdges, yEdges, xBin, yBin, tcounts, pval, mi, pos_align, PSAbool_align,
                 speed_sm, isrunning, cmperbin, speed_thresh, mouse, arena, day,
                 list_dir, nshuf, sr_image, tmap_sm_shuf):
        self.tmap_us = tmap_us
        self.tmap_sm = tmap_gauss
        self.xrun = xrun
        self.yrun = yrun
        self.PSAboolrun = PSAboolrun
        self.nneurons = PSAboolrun.shape[0]
        self.occmap = occmap
        self.runoccmap = runoccmap
        self.xEdges = xEdges
        self.yEdges = yEdges
        self.xBin = xBin
        self.yBin = yBin
        self.tcounts = tcounts
        self.pval = pval
        self.mi = mi
        self.pos_align = pos_align
        self.PSAbool_align = PSAbool_align
        self.speed_sm = speed_sm
        self.isrunning = isrunning
        self.cmperbin = cmperbin
        self.speed_thresh = speed_thresh
        self.mouse = mouse
        self.arena = arena
        self.day = day
        self.list_dir = list_dir
        self.nshuf = nshuf
        self.sr_image = sr_image
        self.tmap_sm_shuf = tmap_sm_shuf

    def save_data(self, save_file):
        with open(save_file, 'wb') as output:
            dump(self, output)

    def pfscroll(self, current_position=0, pval_thresh=1.01, plot_xy=False, link_PFO=None):
        """Scroll through placefields with trajectory + firing in one plot, smoothed tmaps in another subplot,
        and unsmoothed tmaps in another

        :param current_position: index in spatially tuned neuron ndarray to start with (clunky, since you don't
        know how many spatially tuned neurons you have until you threshold them below).
        :param pval_thresh: default = 1. Only scroll through neurons with pval (based on mutual information scores
        calculated after circularly permuting calcium traces/events) < pval_thresh
        :param plot_xy: plot x and y position versus time with calcium activity indicated in red.
        :param link_PFO: placefield object to link to for matched scrolling.
        :return:
        """

        # Get only spatially tuned neurons: those with mutual spatial information pval < pval_thresh
        if self.nshuf > 0:
            spatial_neurons = np.where([a < pval_thresh for a in self.pval])[0]
        elif self.nshuf == 0:
            spatial_neurons = np.arange(0, self.nneurons, 1)

        # Plot frame and position of mouse.
        titles = ["Neuron " + str(n) for n in spatial_neurons]  # set up array of neuron numbers

        # Hijack Will's ScrollPlot function to scroll through each neuron
        lims = [[self.xEdges.min(), self.xEdges.max()], [self.yEdges.min(), self.yEdges.max()]]
        if not plot_xy:
            self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm),
                                current_position=current_position, n_neurons=len(spatial_neurons),
                                n_rows=1, n_cols=3, figsize=(17.2, 5.3), titles=titles,
                                x=self.pos_align[0, self.isrunning], y=self.pos_align[1, self.isrunning],
                                traj_lims=lims, PSAbool=self.PSAboolrun[spatial_neurons, :],
                                tmap_us=[self.tmap_us[a] for a in spatial_neurons],
                                tmap_sm=[self.tmap_sm[a] for a in spatial_neurons],
                                mouse=self.mouse, arena=self.arena, day=self.day, link_obj=link_PFO)
        elif plot_xy:
            # quick bugfix - earlier versions of Placefields spit out sr_image as multi-level list
            sr_image = self.sr_image
            while type(sr_image) is not int:
                sr_image = sr_image[0]

            self.f = ScrollPlot((plot_events_over_pos, plot_tmap_us, plot_tmap_sm, plot_psax, plot_psay),
                                current_position=current_position, n_neurons=len(spatial_neurons),
                                n_rows=3, n_cols=3, combine_rows=[1, 2], figsize=(12.43, 9.82), titles=titles,
                                x=self.pos_align[0, self.isrunning], y=self.pos_align[1, self.isrunning],
                                traj_lims=lims, PSAbool=self.PSAboolrun[spatial_neurons, :],
                                tmap_us=[self.tmap_us[a] for a in spatial_neurons],
                                tmap_sm=[self.tmap_sm[a] for a in spatial_neurons],
                                mouse=self.mouse, arena=self.arena, day=self.day, sample_rate=sr_image,
                                link_obj=link_PFO)



class ScrollPlot:
    """
    Plot stuff then scroll through it! A bit hacked together as of 2/28/2020. Better would be to input a figure and axes
    along with the appropriate plotting functions?

    :param plot_func: tuple of plotting functions to plot into the appropriate axes
    :param x: X axis data.
    :param y: Y axis data.
    :param xlabel = 'x': X axis label.
    :param ylabel = 'y': Y axis label.
    :param link_obj = if specified, you can link to another object and scroll side-by-side
        with one key-press, None(default) = keep un-linked.
    :param combine_rows = list of subplots rows to combine into one subplot. Currently only supports doing all bottom
        rows which must match the functions specified in plot_func
    """

    # Initialize the class. Gather the data and labels.
    def __init__(self, plot_func, xlabel='', ylabel='',
                 titles=([' '] * 10000), n_rows=1,
                 n_cols=1, figsize=(8, 6), combine_rows=[],
                 link_obj=None, **kwargs):

        self.plot_func = plot_func
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.titles = titles
        self.n_rows = n_rows  # NK can make default = len(plot_func)
        self.n_cols = n_cols
        self.share_y = False
        self.share_x = False
        self.figsize = figsize

        # Dump all arguments into ScrollPlot.
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.fig, self.ax, = plt.subplots(self.n_rows, self.n_cols,
                                           sharey=self.share_y,
                                           sharex=self.share_x,
                                           figsize=self.figsize)
        if n_cols == 1 and n_rows == 1:
            self.ax = (self.ax,)

        # Make rows into one subplot if specified
        if len(combine_rows) > 0:
            for row in combine_rows:
                plt.subplot2grid((self.n_rows, self.n_cols), (row, 0), colspan=self.n_cols, fig=self.fig)
            self.ax = self.fig.get_axes()

        # Flatten into 1d array if necessary and not done already via combining rows
        if n_cols > 1 and n_rows > 1 and hasattr(self.ax, 'flat'):
             self.ax = self.ax.flat

        # Necessary for scrolling.
        if not hasattr(self, 'current_position'):
            self.current_position = 0

        # Plot the first plot of each function and label
        for ax_ind, plot_f in enumerate(self.plot_func):
            plot_f(self, ax_ind)
            self.apply_labels()
            # print(str(ax_ind))




        # Connect the figure to keyboard arrow keys.
        if link_obj is None:
            self.fig.canvas.mpl_connect('key_press_event',
                                    lambda event: self.update_plots(event))
        else:
            link_obj.fig.canvas.mpl_connect('key_press_event',
                                            lambda event: self.update_plots(event))

    # Go up or down the list. Left = down, right = up.
    def scroll(self, event):
        if event.key == 'right' and self.current_position <= self.last_position:
            if self.current_position <= self.last_position:
                if self.current_position == self.last_position:
                    self.current_position = 0
                else:
                    self.current_position += 1
        elif event.key == 'left' and self.current_position >= 0:
            if self.current_position == 0:
                self.current_position = self.last_position
            else:
                self.current_position -= 1
        elif event.key == '6':
            if (self.current_position + 15) < self.last_position:
                self.current_position += 15
            elif (self.current_position + 15) >= self.last_position:
                if self.current_position == self.last_position:
                    self.current_position = 0
                else:
                    self.current_position = self.last_position
        elif event.key == '4':
            print('current position before = ' + str(self.current_position))
            if self.current_position > 15:
                self.current_position -= 15
            elif self.current_position <= 15:
                if self.current_position == 0:
                    self.current_position = self.last_position
                else:
                    self.current_position = 0
            print('current position after = ' + str(self.current_position))
        elif event.key == '9' and (self.current_position + 100) < self.last_position:
            self.current_position += 100
        elif event.key == '7' and self.current_position > 100:
            self.current_position -= 100

            # Apply axis labels.
    def apply_labels(self):
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.titles[self.current_position])

    # Update the plot based on keyboard inputs.
    def update_plots(self, event):
        # Clear axis.
        try:
            for ax in self.ax:
                ax.cla()
                # print('Cleared axes!')
        except:
            self.ax.cla()

        # Scroll then update plot.
        self.scroll(event)

        # Run the plotting function.
        for ax_ind, plot_f in enumerate(self.plot_func):
            plot_f(self, ax_ind)
            # self.apply_labels()

        # Draw.
        self.fig.canvas.draw()

        if event.key == 'escape':
            plt.close(self.fig)


def neuron_number_title(neurons):
    titles = ["Neuron: " + str(n) for n in neurons]

    return titles