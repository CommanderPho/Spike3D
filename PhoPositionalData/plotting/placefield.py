import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from PhoPositionalData.plotting.saving import save_to_multipage_pdf

def plot_placefield_tuning_curve(xbin_centers, tuning_curve, ax, is_horizontal=False, color='g'):
    """ Plots the 1D Normalized Tuning Curve in a 2D Plot
    Usage:
        axs1 = plot_placefield_tuning_curve(active_epoch_placefields1D.ratemap.xbin_centers, active_epoch_placefields1D.ratemap.normalized_tuning_curves[curr_cell_id, :].squeeze(), axs1)
    """
    if is_horizontal:
        ax.fill_betweenx(xbin_centers, tuning_curve, color=color, alpha=0.3, interpolate=True)
        ax.plot(tuning_curve, xbin_centers, color, alpha=0.8)
    else:
        ax.fill_between(xbin_centers, tuning_curve, color=color, alpha=0.3)
        ax.plot(xbin_centers, tuning_curve, color, alpha=0.8)
    return ax

def _plot_helper_build_jittered_spike_points(curr_cell_spike_times, curr_cell_interpolated_spike_curve_values, jitter_multiplier=2.0, feature_range=(-1, 1), time_independent_jitter=False):
    """ jitters the curve_value for each spike based on the time it occured along the curve or a time_independent positive jitter """
    if time_independent_jitter:
        jitter_add = np.abs(np.random.randn(len(curr_cell_spike_times))) * jitter_multiplier
    else:
        # jitter the spike points based on the time they occured.
        jitter_add = jitter_multiplier * minmax_scale(curr_cell_spike_times, feature_range=feature_range)
    return curr_cell_interpolated_spike_curve_values + jitter_add

def _plot_helper_setup_gridlines(ax, bin_edges, bin_centers):
    ax.set_yticks(bin_edges, minor=False)
    ax.set_yticks(bin_centers, minor=True)
    ax.yaxis.grid(True, which='major', color = 'grey', linewidth = 0.5) # , color = 'green', linestyle = '--', linewidth = 0.5
    ax.yaxis.grid(True, which='minor', color = 'grey', linestyle = '--', linewidth = 0.25)


        
        
def plot_1d_placecell_validations(active_placefields1D, plotting_config, should_save=False, modifier_string='', save_mode='separate_files'):
    """ Uses plot_1D_placecell_validation(...) to plot a series of plots, one for each potential placecell, that allows you to see how the spiking corresponds to the animal's position/lap and how that contributes to the computed placemap
    
    Usage:
        plot_1d_placecell_validations(active_epoch_placefields1D, should_save=True)
        plot_1d_placecell_validations(active_epoch_placefields1D, modifier_string='lap_only', should_save=False)

    """
    # def _filename_for_placefield(active_epoch_placefields1D, curr_cell_id):
    #     return active_epoch_placefields1D.str_for_filename(is_2D=False) + '-cell_{:02d}'.format(curr_cell_id)
    
    n_cells = active_placefields1D.ratemap.n_neurons
    out_figures_list = []
    out_axes_list = []
    
    if should_save:
        curr_parent_out_path = plotting_config.active_output_parent_dir.joinpath('1d Placecell Validation')
        curr_parent_out_path.mkdir(parents=True, exist_ok=True)        
        
    for i in np.arange(n_cells):
        curr_cell_id = active_placefields1D.cell_ids[i]
        fig, axs = plot_1D_placecell_validation(active_placefields1D, i)
        out_figures_list.append(fig)
        out_axes_list.append(axs)

    # once done, save out as specified
    if should_save:
        common_basename = active_placefields1D.str_for_filename(prefix_string=modifier_string)
        if save_mode == 'separate_files':
            # make a subdirectory for this run (with these parameters and such)
            curr_specific_parent_out_path = curr_parent_out_path.joinpath(common_basename)
            curr_specific_parent_out_path.mkdir(parents=True, exist_ok=True)
            print(f'Attempting to write {n_cells} separate figures to {str(curr_specific_parent_out_path)}')
            for i in np.arange(n_cells):
                print('Saving figure {} of {}...'.format(i, n_cells))
                curr_cell_id = active_placefields1D.cell_ids[i]
                fig = out_figures_list[i]
                # curr_cell_filename = 'pf1D-' + modifier_string + _filename_for_placefield(active_placefields1D, curr_cell_id) + '.png'
                curr_cell_basename = '-'.join([common_basename, f'cell_{curr_cell_id:02d}'])
                # add the file extension
                curr_cell_filename = f'{curr_cell_basename}.png'
                active_pf_curr_cell_output_filepath = curr_specific_parent_out_path.joinpath(curr_cell_filename)
                fig.savefig(active_pf_curr_cell_output_filepath)
        elif save_mode == 'pdf':
            print('saving multipage pdf...')
            curr_cell_basename = common_basename
            # add the file extension
            curr_cell_filename = f'{curr_cell_basename}-multipage_pdf.pdf'
            pdf_save_path = curr_parent_out_path.joinpath(curr_cell_filename)
            save_to_multipage_pdf(out_figures_list, save_file_path=pdf_save_path)
        else:
            raise ValueError
        print('\t done.')
    return out_figures_list

# 2d Placefield comparison figure:
def plot_1D_placecell_validation(active_epoch_placefields1D, placefield_cell_index):
    """ A single cell method of analyzing 1D placefields and the spikes that create them 
    
    placefield_cell_index: an flat index into active_epoch_placefields1D.cell_ids. Must be between 0 and len(active_epoch_placefields1D.cell_ids). NOT the cell's original ID!
    """
    
    curr_cell_id = active_epoch_placefields1D.cell_ids[placefield_cell_index]
    # jitter the curve_value for each spike based on the time it occured along the curve:
    jitter_multiplier = 0.05
    # feature_range = (-1, 1)
    feature_range = (0, 1)
    should_plot_spike_indicator_points_on_placefield = True
    should_plot_spike_indicator_lines_on_trajectory = True
    spike_indicator_lines_alpha = 1.0
    spike_indcator_lines_linewidth = 0.3
    should_plot_bins_grid = False

    fig = plt.figure(figsize=(23, 9.7))
    # fig.set_size_inches([23, 9.7])
    # Layout Subplots in Figure:
    gs = fig.add_gridspec(1, 8)
    gs.update(wspace=0, hspace=0.05) # set the spacing between axes.
    axs0 = fig.add_subplot(gs[0, :-1])
    axs1 = fig.add_subplot(gs[0, -1], sharey=axs0)
    axs1.set_title('Normalized Placefield', fontsize='14')
    axs1.set_xticklabels([])
    axs1.set_yticklabels([])

    ## The main position vs. spike curve:
    active_epoch_placefields1D.plotRaw_v_time(placefield_cell_index, ax=axs0)
    
    # Title and Subtitle:
    title_string = ' '.join(['pf1D', f'Cell {curr_cell_id:02d}'])
    subtitle_string = ' '.join([f'{active_epoch_placefields1D.config.str_for_display(False)}'])
    fig.suptitle(title_string, fontsize='22')
    axs0.set_title(subtitle_string, fontsize='16')
    
    # axs0.yaxis.grid(True, color = 'green', linestyle = '--', linewidth = 0.5)
    if should_plot_bins_grid:
        _plot_helper_setup_gridlines(axs0, active_epoch_placefields1D.ratemap.xbin, active_epoch_placefields1D.ratemap.xbin_centers)


    ## Part 2: The Placefield Plot to the Right and the connecting features:
    ## The individual spike lines:
    curr_cell_spike_times = active_epoch_placefields1D.ratemap_spiketrains[placefield_cell_index]  # (271,)
    curr_cell_spike_positions = active_epoch_placefields1D.ratemap_spiketrains_pos[placefield_cell_index]  # (271,)
    curr_cell_normalized_tuning_curve = active_epoch_placefields1D.ratemap.normalized_tuning_curves[placefield_cell_index, :].squeeze()

    # Interpolate the tuning curve for all the spike values:
    curr_cell_interpolated_spike_positions = np.interp(curr_cell_spike_positions, active_epoch_placefields1D.ratemap.xbin_centers, active_epoch_placefields1D.ratemap.xbin_centers) # (271,)
    curr_cell_interpolated_spike_curve_values = np.interp(curr_cell_spike_positions, active_epoch_placefields1D.ratemap.xbin_centers, curr_cell_normalized_tuning_curve) # (271,)
    curr_cell_jittered_spike_curve_values = _plot_helper_build_jittered_spike_points(curr_cell_spike_times, curr_cell_interpolated_spike_curve_values,
                                                                                     jitter_multiplier=jitter_multiplier, feature_range=feature_range, time_independent_jitter=False)
    if should_plot_spike_indicator_lines_on_trajectory:
        # plot the orange lines that span across the position plot to the right
        axs0.hlines(y=curr_cell_interpolated_spike_positions, xmin=curr_cell_spike_times, xmax=curr_cell_spike_times[-1],
                    linestyles='solid', color='orange', alpha=spike_indicator_lines_alpha, linewidth=spike_indcator_lines_linewidth) # plot the lines that underlie the spike points
    axs0.set_xlim((np.min(curr_cell_spike_times), np.max(curr_cell_spike_times)))

    ## The computed placefield on the right-hand side:
    axs1 = plot_placefield_tuning_curve(active_epoch_placefields1D.ratemap.xbin_centers, curr_cell_normalized_tuning_curve, axs1, is_horizontal=True)
    if should_plot_spike_indicator_points_on_placefield:
        axs1.hlines(y=curr_cell_interpolated_spike_positions, xmin=np.zeros_like(curr_cell_jittered_spike_curve_values), xmax=curr_cell_jittered_spike_curve_values, linestyles='solid', color='orange', alpha=spike_indicator_lines_alpha, linewidth=spike_indcator_lines_linewidth) # plot the lines that underlie the spike points
        # axs1.hlines(y=curr_cell_interpolated_spike_positions, xmin=curr_cell_interpolated_spike_curve_values, xmax=curr_cell_jittered_spike_curve_values, linestyles='solid', color='orange', alpha=1.0, linewidth=0.25) # plot the lines that underlie the spike points
        axs1.scatter(curr_cell_jittered_spike_curve_values, curr_cell_interpolated_spike_positions, c='r', marker='_', alpha=0.5) # plot the points themselves
    axs1.axis('off')
    axs1.set_xlim((0, 1))
    axs1.set_ylim((-72, 150))
    return fig, [axs0, axs1]
