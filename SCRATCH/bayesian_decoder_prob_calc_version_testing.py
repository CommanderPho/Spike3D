"""
title: bayesian_decoder_prob_calc_version_testing.py
date: 2023-04-05 11:57:13

Tests several variants of `neuropy_bayesian_prob`
"""

import numpy as np
import pandas as pd
from scipy.special import factorial, logsumexp
import matplotlib.pyplot as plt

# ## Flat common term
# np.exp(-tau * np.sum(F,axis=0)) # (n_pos_bins, )

# np.squeeze(P_x) * np.exp(-tau * np.sum(F,axis=0)) # (n_pos_bins, )

# # Capture test parameters:
# test_parameters = {'tau':tau, 'P_x':P_x, 'F':F, 'n':n}


# # To load test parameters:
# with np.load('test_parameters-neuropy_bayesian_prob.npz') as npzfile:
#         tau = npzfile['tau']
#         P_x = npzfile['P_x']
#         F = npzfile['F']
#         n = npzfile['n']


# # To save test parameters:
# np.savez_compressed('test_parameters-neuropy_bayesian_prob.npz', **{'tau':tau, 'P_x':P_x, 'F':F, 'n':n})

# np.savez_compressed('test_parameters-neuropy_bayesian_prob.npz', **{'tau':time_bin_size, 'P_x':self.P_x, 'F':self.F, 'n':unit_specific_time_binned_spike_counts})


def neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=True, debug_print=False):
    """ 
        n_i: the number of spikes fired by each cell during the time window of consideration
        use_flat_computation_mode: bool - if True, a more memory efficient accumulating computation is performed that avoids `MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64` caused by allocating the full `cell_prob` matrix

    NOTES: Flat vs. Full computation modes:
    Originally 
        cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) 
    This was updated throughout the loop, and then after the loop completed np.prod(cell_prob, axis=2) was used to collapse along axis=2 (nCells), leaving the output posterior with dimensions (nFlatPositionBins, nTimeBins)

    To get around this, I introduced a version that accumulates the multilications over the course of the loop.
        cell_prob = np.ones((nFlatPositionBins, nTimeBins))

    Note: This means that the "Flat" implementation may be more susceptible to numerical underflow, as the intermediate products can become very small, whereas the "Full" implementation does not have this issue. However, the "Flat" implementation can be more efficient in terms of memory usage and computation time, as it avoids creating a large number of intermediate arrays.

    """
    assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'        
    if debug_print:
        print(f'np.shape(P_x): {np.shape(P_x)}, np.shape(F): {np.shape(F)}, np.shape(n): {np.shape(n)}')
    # np.shape(P_x): (1066, 1), np.shape(F): (1066, 66), np.shape(n): (66, 3530)
    
    nCells = n.shape[0]
    nTimeBins = n.shape[1] # many time_bins
    nFlatPositionBins = np.shape(P_x)[0]

    F = F.T # Transpose F so it's of the right form
    
    if use_flat_computation_mode:
        ## Single-cell flat version which updates each iteration:
        cell_prob = np.ones((nFlatPositionBins, nTimeBins)) # Must start with ONES (not Zeros) since we're accumulating multiplications

    else:
        # Full Version which leads to MemoryError when nCells is too large:
        cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) ## MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64

    for cell in range(nCells):
        """ Comparing to the Zhang paper: the output posterior is P_n_given_x (Eqn 35)
            cell_ratemap: [f_{i}(x) for i in range(nCells)]
            cell_spkcnt: [n_{i} for i in range(nCells)]            
        """
        cell_spkcnt = n[cell, :][np.newaxis, :] # .shape: (1, nTimeBins)
        cell_ratemap = F[cell, :][:, np.newaxis] # .shape: (nFlatPositionBins, 1)
        coeff = 1.0 / (factorial(cell_spkcnt)) # 1/factorial(n_{i}) term # .shape: (1, nTimeBins)

        if use_flat_computation_mode:
            # Single-cell flat Version:
            cell_prob *= (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap)) # product equal using *=
            # cell_prob.shape (nFlatPositionBins, nTimeBins)
        else:
            # Full Version:
            # broadcasting
            cell_prob[:, :, cell] = (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap))

    if use_flat_computation_mode:
        # Single-cell flat Version:
        posterior = cell_prob # The product has already been accumulating all along
        posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n
    else:
        # Full Version:
        posterior = np.prod(cell_prob, axis=2) # note this product removes axis=2 (nCells)
        posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n

    return posterior



def neuropy_bayesian_prob_simplified(tau, P_x, F, n, debug_print=False):
    """ Attempt to simplify the neuropy_bayesian_prob function 
        n_i: the number of spikes fired by each cell during the time window of consideration
        use_flat_computation_mode: bool - if True, a more memory efficient accumulating computation is performed that avoids `MemoryError: Unable to allocate 65.4 GiB for an array with shape (3969, 21896, 101) and data type float64` caused by allocating the full `cell_prob` matrix

    NOTES: Flat vs. Full computation modes:
    Originally 
        cell_prob = np.zeros((nFlatPositionBins, nTimeBins, nCells)) 
    This was updated throughout the loop, and then after the loop completed np.prod(cell_prob, axis=2) was used to collapse along axis=2 (nCells), leaving the output posterior with dimensions (nFlatPositionBins, nTimeBins)

    To get around this, I introduced a version that accumulates the multilications over the course of the loop.
        cell_prob = np.ones((nFlatPositionBins, nTimeBins))

    Note: This means that the "Flat" implementation may be more susceptible to numerical underflow, as the intermediate products can become very small, whereas the "Full" implementation does not have this issue. However, the "Flat" implementation can be more efficient in terms of memory usage and computation time, as it avoids creating a large number of intermediate arrays.

    """
    assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'        
    if debug_print:
        print(f'np.shape(P_x): {np.shape(P_x)}, np.shape(F): {np.shape(F)}, np.shape(n): {np.shape(n)}')
    # np.shape(P_x): (1066, 1), np.shape(F): (1066, 66), np.shape(n): (66, 3530)
    
    nCells = n.shape[0]
    nTimeBins = n.shape[1] # many time_bins
    nFlatPositionBins = np.shape(P_x)[0]

    F = F.T # Transpose F so it's of the right form
    
    cell_prob = np.ones((nFlatPositionBins, nTimeBins)) # Must start with ONES (not Zeros) since we're accumulating multiplications

    for cell in range(nCells):
        """ Comparing to the Zhang paper: the output posterior is P_n_given_x (Eqn 35)
            cell_ratemap: [f_{i}(x) for i in range(nCells)]
            cell_spkcnt: [n_{i} for i in range(nCells)]            
        """
        cell_spkcnt = n[cell, :][np.newaxis, :] # .shape: (1, nTimeBins)
        cell_ratemap = F[cell, :][:, np.newaxis] # .shape: (nFlatPositionBins, 1)
        coeff = 1.0 / (factorial(cell_spkcnt)) # 1/factorial(n_{i}) term # .shape: (1, nTimeBins)

        cell_prob *= (((tau * cell_ratemap) ** cell_spkcnt) * coeff) * (np.exp(-tau * cell_ratemap)) # product equal using *=
        # cell_prob.shape (nFlatPositionBins, nTimeBins)
        

    posterior = cell_prob # The product has already been accumulating all along
    posterior /= np.sum(posterior, axis=0) # C(tau, n) = np.sum(posterior, axis=0): normalization condition mentioned in eqn 36 to convert to P_x_given_n
    
    return posterior


def neuropy_bayesian_prob_improved(tau, P_x, F, n):
    """ Log-probability version suggested by ChatGPT. Needs to be tested.
    n_i: the number of spikes fired by each cell during the time window of consideration

    NOTES: Logarithmic probability version:
    This version uses logarithms of probabilities to avoid numerical underflow and memory errors

    ChatGPT: This implementation computes the logarithm of the probability of each cell spiking given a position bin and time bin, and then sums the logarithms for each cell to get the total logarithm of the probability of all cells spiking. It then uses the log-sum-exp trick to compute the posterior probability of the position given the spike counts, and converts the logarithmic posterior probabilities back to normal probabilities. This approach should avoid both numerical underflow and memory errors, and should be more efficient than the full version since it only stores logarithms of probabilities instead of probabilities themselves.

    """
    assert(len(n) == np.shape(F)[1]), f'n must be a column vector with an entry for each place cell (neuron). Instead it is of np.shape(n): {np.shape(n)}. np.shape(F): {np.shape(F)}'
    nCells = n.shape[0]
    nTimeBins = n.shape[1] # many time_bins
    nPositionBins = np.shape(P_x)[0]

    # Transpose F so it's of the right form
    F = F.T

    # Compute the log probability of each cell spiking given a position bin and time bin
    log_cell_prob = np.zeros((nPositionBins, nTimeBins, nCells))
    for cell in range(nCells):
        # Extract the spike counts and firing rates for the current cell
        cell_spkcnt = n[cell, :][np.newaxis, :] # .shape: (1, nTimeBins)
        cell_ratemap = F[cell, :][:, np.newaxis] # .shape: (nPositionBins, 1)

        # Compute the logarithm of the coefficient 1/factorial(n_i) for each time bin
        log_coeff = -np.log(factorial(cell_spkcnt))

        # Compute the logarithm of the probability of the cell spiking at each time bin for each position bin
        log_cell_prob[:, :, cell] = np.log(cell_ratemap * tau) * cell_spkcnt + log_coeff - cell_ratemap * tau

    # Sum the log probabilities for each cell to get the total log probability of all cells spiking
    log_total_prob = np.sum(log_cell_prob, axis=2)

    # Compute the posterior probability of the position given the spike counts using the log-sum-exp trick to avoid numerical underflow
    log_posterior = log_total_prob + np.log(P_x)
    log_C_tn = logsumexp(log_posterior, axis=0) # looks like we're doing double log here
    log_P_x_given_n = log_posterior - log_C_tn

    # Convert the logarithmic posterior probabilities back to normal probabilities
    posterior = np.exp(log_P_x_given_n)

    return posterior


def test_neuropy_bayesian_prob(tau, P_x, F, n):

    is_cell_firing = (n > 0)
    num_cells_active =  np.sum(is_cell_firing, axis=0)    
    print(f'num_cells_active: {num_cells_active}')

    # Test the full computation mode
    posterior_full = neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=False)
    
    # Test the flat computation mode
    posterior_flat = neuropy_bayesian_prob(tau, P_x, F, n, use_flat_computation_mode=True)
    
    posterior_simplified = neuropy_bayesian_prob_simplified(tau, P_x, F, n)

    # Test the improved computation mode
    # posterior_improved = neuropy_bayesian_prob_improved(tau, P_x, F, n) # (63, 5)
    # print(f'{posterior_improved.shape = }')

    # ## Plot the outputs for visual comparison
    # import matplotlib.pyplot as plt

    intermediate_term = [(np.exp(-tau * cell_ratemap)) for cell_ratemap in F.T]
    intermediate_term = np.array(intermediate_term)
    print(f'{intermediate_term.shape = }')    
    # intermediate_term = np.sum(intermediate_term, axis=0)
    # intermediate_term = np.prod(intermediate_term, axis=0)
    print(f'{intermediate_term.shape = }')
    # intermediate_term = np.reshape(intermediate_term, (63, 1))
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    axs.imshow(intermediate_term)
    plt.show()

    # fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    # axs[0, 0].imshow(posterior_flat)
    # axs[0, 0].set_title("Flat")
    # axs[0, 1].imshow(posterior_full)
    # axs[0, 1].set_title("Full")
    # axs[1, 0].imshow(posterior_improved)
    # axs[1, 0].set_title("Improved")
    # axs[1, 1].imshow(np.abs(posterior_full - posterior_improved))
    # axs[1, 1].set_title("Difference")
    # plt.show()


    # Ensure that all versions of the function produce the same output
    assert np.allclose(posterior_full, posterior_flat)
    assert np.allclose(posterior_full, posterior_simplified)
    assert np.allclose(posterior_flat, posterior_simplified)

    # assert np.allclose(posterior_full, posterior_improved)
    # assert np.allclose(posterior_flat, posterior_improved)
    
    return "All versions of the function produce the same output."


if __name__ == "__main__":
    # To load test parameters:
    # load_path = r"C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\test_parameters-neuropy_bayesian_prob.npz"
    load_path = "/home/halechr/repos/NeuroPy/tests/test_parameters-neuropy_bayesian_prob.npz"
    # load_path = 'test_parameters-neuropy_bayesian_prob.npz'
    with np.load(load_path) as npzfile:
            tau = npzfile['tau']
            P_x = npzfile['P_x']
            F = npzfile['F']
            n = npzfile['n']

    print(f'n: {n}')
    test_neuropy_bayesian_prob(tau, P_x, F, n)
