from pathlib import Path
from pyphocorehelpers.print_helpers import DocumentationFilePrinter, print_keys_if_possible


doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='NeuropyPipeline')
doc_printer.save_documentation('NeuropyPipeline', curr_active_pipeline, non_expanded_item_keys=['stage','_reverse_cellID_index_map', 'pf_listed_colormap', 'computation_results', 'active_configs', 'logger', 'plot', '_plot_object'],
                               additional_excluded_item_classes=["<class 'pyphoplacecellanalysis.General.Pipeline.Stages.Display.Plot'>"], max_depth=16) # 'Logger'

