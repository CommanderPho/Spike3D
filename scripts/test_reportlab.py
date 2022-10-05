from glob import glob
from pathlib import Path

## reportlab:
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

## PyPDF2:
from PyPDF2 import PdfMerger, PdfFileMerger

# Common:
root_path = Path(r'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\Screenshots\ProgrammaticDisplayFunctionTesting\2022-10-04')

filenames = ['kdiba_2006-6-07_11-26-53_maze_PYR__display_plot_decoded_epoch_slices_laps_128.pdf', 'kdiba_2006-6-07_11-26-53_maze1_PYR__display_plot_decoded_epoch_slices_laps_128.pdf', 'kdiba_2006-6-07_11-26-53_maze2_PYR__display_plot_decoded_epoch_slices_laps_128.pdf']
filepaths = [root_path.joinpath(a_filename) for a_filename in filenames]


def main_pypdf2(allpdfs, outpath=None):
    """ works to merge multiple .pdf files specified by allpdfs into a single output pdf. 

    See https://stackoverflow.com/questions/3444645/merge-pdf-files for more info.
    https://github.com/mahaguru24/Python_Merge_PDF


    output: Path or str - to get all pdfs in a directory as a list: `allpdfs = [a for a in glob("*.pdf")]`
    """
    if outpath is None:
        outpath = 'merged_pdfs.pdf'
    merger = PdfFileMerger()
    [merger.append(pdf) for pdf in allpdfs]
    with open(outpath, "wb") as new_file:
        merger.write(new_file)
    merger.close()


def main():
    c = canvas.Canvas('report.pdf', pagesize=letter)
    # c.drawImage('filename1.png', 0,0)
    # c.drawImage('filename2.png', 0,100)
    c.drawImage(filepaths[0], 0,0)
    c.drawImage(filepaths[1], 0,100)

    c.save() 


# main()

out_pdf = root_path.joinpath('merged_pdfs.pdf')
main_pypdf2(filepaths, outpath=out_pdf)
print(f'done!')
