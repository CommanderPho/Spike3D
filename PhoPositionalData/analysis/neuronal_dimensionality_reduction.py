# neuronal_dimensionality_reduction.py
## Does PCA and ICA to generate 2D plots
import sys
import numpy as np
import matplotlib.pyplot as plt
from neuropy import core


def runAnalysis_PCAandICA(active_session_Neurons, bin_size=0.250, frate_thresh=0.1, should_plot=False, active_cells_colormap=None):
    """ ## ICA and PCA Analysis
    
    Usage:
        should_show_2D_ICA_plots = False
        from PhoPositionalData.analysis.neuronal_dimensionality_reduction import runAnalysis_PCAandICA
        active_session_ensembles, template, zsc_template, pca_data = runAnalysis_PCAandICA(active_epoch_session.neurons, bin_size=0.250, frate_thresh=0.1, should_plot=should_show_2D_ICA_plots, active_cells_colormap=active_config.plotting_config.active_cells_colormap)

    """
    ## TODO: not sure if I want plotting in here, this was just factored out of the Jupyter-lab notebook
    from neuropy.analyses.reactivation import NeuronEnsembles
    # active_epoch_placefields.frate_thresh
    active_session_ensembles = NeuronEnsembles(active_session_Neurons, bin_size=bin_size, frate_thresh=frate_thresh)
    active_neuron_ids_include_mask = active_session_ensembles.neuron_included_indx_thresh
    if should_plot:
        # show the ensembles plot:
        active_session_ensembles.plot_ensembles()

    ## Activations:
    active_session_ensembles.calculate_activation()
    
    if should_plot:
        active_session_ensembles.plot_activation()

    # print('n_ensembles: {}'.format(active_session_ensembles.n_ensembles))
    # print('np.shape(weights): {}'.format(np.shape(active_session_ensembles.weights))) # shape (32, 8)
    # np.shape(active_neuron_ids_include_mask) # (33,)
    # np.shape(active_cells_colormap) # (33, 4)
    
    # Build the input matrix required for PCA
    template, zsc_template = active_session_ensembles.get_original_data() # np.shape(template): (32, 12992)
    # print('np.shape(template): {}, np.shape(zsc_template): {}'.format(np.shape(template), np.shape(zsc_template)))
    if should_plot:
        plt.figure()
        # plt.scatter(zsc_template[:, 0], zsc_template[:, 1], c=active_cells_colormap[active_neuron_ids_include_mask], s=30)
        plt.scatter(template[:, 0], template[:, 1], c=active_cells_colormap[active_neuron_ids_include_mask], s=30)
        plt.title('Original Neuronal Data prior to PCA Decomposition')
        plt.show()

    
    from sklearn import decomposition
    pca = decomposition.PCA()
    pca.n_components = 2 # project onto two dimensions
    pca_data = pca.fit_transform(zsc_template)
    # print('np.shape(pca_data): {}'.format(np.shape(pca_data))) # np.shape(pca_data): (32, 2)
    if should_plot:
        plt.figure()
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=active_cells_colormap[active_neuron_ids_include_mask, :], s=30)
        plt.title('PCA Decomposition for the 2 largest components')
        plt.show()
 
    return active_session_ensembles, template, zsc_template, pca_data


