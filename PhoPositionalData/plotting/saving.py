import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

""" 
# Simple Figure Saving Demonstration:

aFig.suptitle('Lap Trajectories 2D', fontsize=22)
fig_out_path = active_config.plotting_config.get_figure_save_path('lap_trajectories_2D').with_suffix('.png')
aFig.savefig(fig_out_path)


"""
# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
def save_to_multipage_pdf(figures_list, save_file_path='multipage_pdf.pdf'):
    """Saves out a list of figures to a multi-page PDF file on disk. One figure is exported per page.
    """
    num_figures_to_save = len(figures_list)
    with PdfPages(save_file_path) as pdf:
        print('trying to save multipage PDF of {} figures to {}...'.format(num_figures_to_save, str(save_file_path)), end='')
        plt.rc('text', usetex=False)
        for a_figure in figures_list:
            pdf.savefig(a_figure)
            
        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = 'Multipage PDF Example'
        d['Author'] = 'Pho Hale'
        d['Subject'] = 'How to create a multipage pdf file and set its metadata'
        d['Keywords'] = 'PdfPages multipage keywords author title subject'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()
        print('\t done.')

        
def save_figure(figures_list, plotting_config, common_basename, save_mode='separate_files'):
    n_figures = len(figures_list)
    curr_parent_out_path = plotting_config.active_output_parent_dir.joinpath('1d Placecell Validation')
    curr_parent_out_path.mkdir(parents=True, exist_ok=True)
    # once done, save out as specified
    if save_mode == 'separate_files':
        # make a subdirectory for this run (with these parameters and such)
        curr_specific_parent_out_path = curr_parent_out_path.joinpath(common_basename)
        curr_specific_parent_out_path.mkdir(parents=True, exist_ok=True)
        print(f'Attempting to write {n_figures} separate figures to {str(curr_specific_parent_out_path)}')
        for i in np.arange(n_figures):
            print('Saving figure {} of {}...'.format(i, n_figures))
            # curr_cell_id = active_placefields1D.cell_ids[i]
            fig = figures_list[i]
            # curr_specific_basename = '-'.join([common_basename, f'cell_{curr_cell_id:02d}'])
            # add the file extension
            curr_fig_filename = f'{curr_specific_basename}.png'
            active_pf_curr_cell_output_filepath = curr_specific_parent_out_path.joinpath(curr_fig_filename)
            fig.savefig(active_pf_curr_cell_output_filepath)
    elif save_mode == 'pdf':
        print('saving multipage pdf...')
        curr_specific_basename = common_basename
        # add the file extension
        curr_fig_filename = f'{curr_specific_basename}-multipage_pdf.pdf'
        pdf_save_path = curr_parent_out_path.joinpath(curr_fig_filename)
        save_to_multipage_pdf(figures_list, save_file_path=pdf_save_path)
    else:
        raise ValueError
    print('\t done.')