# 2023-03-08 - Build Transition Matrix for each position bin
from copy import deepcopy
import numpy as np
from neuropy.utils.mixins.binning_helpers import transition_matrix


### 1D Transition Matrix:
pf1D = deepcopy(curr_active_pipeline.computation_results['maze1'].computed_data['pf1D'])
pf1D.bin_info

num_position_states = len(pf1D.xbin_labels)
binned_x = pf1D.filtered_pos_df['binned_x'].to_numpy()
binned_x_indicies = binned_x - 1
binned_x_transition_matrix = transition_matrix(deepcopy(binned_x_indicies), markov_order=1, max_state_index=num_position_states)
# binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, transition_matrix(deepcopy(binned_x_indicies), markov_order=2, max_state_index=num_position_states), transition_matrix(deepcopy(binned_x_indicies), markov_order=3, max_state_index=num_position_states)]

binned_x_transition_matrix[np.isnan(binned_x_transition_matrix)] = 0.0
binned_x_transition_matrix_higher_order_list = [binned_x_transition_matrix, np.linalg.matrix_power(binned_x_transition_matrix, 2), np.linalg.matrix_power(binned_x_transition_matrix, 3)]

binned_x_transition_matrix.shape # (64, 64)





# # Visualization ______________________________________________________________________________________________________ #
# from pyphoplacecellanalysis.GUI.PyQtPlot.BinnedImageRenderingWindow import BasicBinnedImageRenderingWindow, LayoutScrollability
# out = BasicBinnedImageRenderingWindow(binned_x_transition_matrix, pf1D.xbin_labels, pf1D.xbin_labels, name='binned_x_transition_matrix', title="Transition Matrix for binned x (from, to)", variable_label='Transition Matrix', scrollability_mode=LayoutScrollability.NON_SCROLLABLE)
# # out.add_data(row=1, col=0, matrix=active_eloy_analysis.pf_overlapDensity_2D, xbins=active_pf_2D_dt.xbin_labels, ybins=active_pf_2D_dt.ybin_labels, name='pf_overlapDensity', title='pf overlapDensity metric', variable_label='pf overlapDensity')

# out.add_data(row=1, col=0, matrix=binned_x_transition_matrix_higher_order_list[1], xbins=pf1D.xbin_labels, ybins=pf1D.ybin_labels, name='binned_x_transition_matrix^2', title='2nd Order Transition Matrix for binned x (from, to)', variable_label='2nd Order Transition Matrix') # , scrollability_mode=LayoutScrollability.NON_SCROLLABLE
# out.add_data(row=2, col=0, matrix=binned_x_transition_matrix_higher_order_list[2], xbins=pf1D.xbin_labels, ybins=pf1D.ybin_labels, name='binned_x_transition_matrix^3', title='3rd Order Transition Matrix for binned x (from, to)', variable_label='3rd Order Transition Matrix') # , scrollability_mode=LayoutScrollability.NON_SCROLLABLE

# out2 = BasicBinnedImageRenderingWindow(binned_x_transition_matrix_higher_order_list[1], pf1D.xbin_labels, pf1D.xbin_labels, name='binned_x_transition_matrix^2', title="2nd Order Transition Matrix for binned x (from, to)", variable_label='2nd Order Transition Matrix', scrollability_mode=LayoutScrollability.NON_SCROLLABLE)

# out3 = BasicBinnedImageRenderingWindow(binned_x_transition_matrix_higher_order_list[2], pf1D.xbin_labels, pf1D.xbin_labels, name='binned_x_transition_matrix^3', title="3rd Order Transition Matrix for binned x (from, to)", variable_label='3rd Order Transition Matrix', scrollability_mode=LayoutScrollability.NON_SCROLLABLE)

