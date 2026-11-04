import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from collections import defaultdict
from datetime import datetime
from pathlib import Path


class PhoDibaPaperVectorCraftHelper:

	figs_layer_geometry = {
		'Artboard 3.pdf': {
			# 'Fig3a+b': {'x': 5.7407, 'y': 7.447218750000047, 'width': 129.1757, 'height': 241.66897968749993},
			'Fig3a.svg': {'x': 9.3114859375, 'y': 11.197580273437529, 'width': 125.6046140625, 'height': 114.23601816406247},
			'Fig3b.svg': {'x': 9.3114859375, 'y': 134.06868027343748, 'width': 125.60491406250001, 'height': 115.0475181640625},
			# 'fig3_c_data_diag_hist': {'x': 398.5329, 'y': 1.0249999999999773, 'width': 67.6481, 'height': 67.64910000000006},
			# 'C_Data.svg': {'x': 201.5388, 'y': 26.771000000000072, 'width': 242.0228, 'height': 214.85709999999995}, # #TODO 2026-10-29 16:01: - [ ] REAL C_Data
			# 'C_Data.svg': {'x': 157.8184, 'y': 8.176718749999964, 'width': 270.5089, 'height': 260.49027343750004}, #TODO 2026-10-29 16:01: - [ ] Fake C_Data from `StaticCParts`
			'C_Data.svg': {'x': 195.46821037898118, 'y': 13.974207317621463, 'width': 234.9447490297146, 'height': 230.33770365816957}, # #TODO 2026-10-30 10:54: - [ ] Modified better boundaries placement
			# 'StaticCParts': {'x': 157.8184, 'y': 8.176718749999964, 'width': 270.5089, 'height': 260.49027343750004},
			# 'Fig3c_data_diag_hist.svg': {'x': 398.5329, 'y': 1.0249999999999773, 'width': 67.6481, 'height': 67.64910000000006},
		},
		'Artboard 4.pdf': {
			'ScatterOverTime_Laps_perTimeBin.svg': {'x': 0.7921446789572979, 'y': 0.0, 'width': 388.47986937169003, 'height': 156.96156338250103},
			'Figure 4_SummaryStats_Laps_perTimeBin.svg': {'x': 369.45444366506405, 'y': 7.341050359617976, 'width': 99.67588609310917, 'height': 141.4210368639443},
			'ScatterOverTime_PBEs_perTimeBin.svg': {'x': 0.5088833995630582, 'y': 150.66224712966167, 'width': 388.47986937169, 'height': 156.961563382501},
			'Figure 4_SummaryStats_PBEs_perTimeBin.svg': {'x': 369.45444366506405, 'y': 159.06777769089888, 'width': 99.67680000000001, 'height': 141.4223335231983},
		},
		'figS3_TimeBinDurationOnDecodingPerform.pdf': {
			'ScatterOverTime_Laps_perTimeBin.svg': {'x': 0.0, 'y': 0.0, 'width': 476.09292351679767, 'height': 190.4410091300099},
			'ScatterOverTime_PBEs_perTimeBin.svg': {'x': -8.432479962904523, 'y': 163.63724948639202, 'width': 484.52540347970216, 'height': 224.26157184586037},
			'Figure S3_SummaryStats_perTimeBin_by_time_bin_size.svg': {'x': 3.2553841573510987, 'y': 351.3835582135365, 'width': 284.59136092010016, 'height': 224.6164417864635},

			# 'ScatterOverTime_Laps_perTimeBin.svg_PROGRAMMATIC': {'x': 2.3713450469581594, 'y': 0.948557143849726, 'width': 471.3502334228814, 'height': 188.5438948423104},
			# 'ScatterOverTime_PBEs_perTimeBin.svg_PROGRAMMATIC': {'x': -8.432479962904514, 'y': 157.86100084746104, 'width': 484.52540347970216, 'height': 193.81406912372236},
			# 'Figure S3_SummaryStats_perTimeBin_by_time_bin_size.svg_PROGRAMMATIC': {'x': 11.980259810880275, 'y': 332.6181027247877, 'width': 271.7651846403536, 'height': 209.79829413492132},

		},
		'Figure S8 - Effect of Heuristic-filtering perTimeBin.pdf': {
			'ScatterOverTime_Laps_perTimeBin.svg': {'x': 0.0, 'y': 0.0, 'width': 468.27133896419633, 'height': 189.20054099563487},
			'ScatterOverTime_PBEs_perTimeBin.svg': {'x': 0.5893954560219754, 'y': 185.72098141999123, 'width': 467.69220359180815, 'height': 188.96654690578106},
			'kdiba_4_Figure S8-pre_post_delta_effect-P_Short.svg': {'x': 6.4321670348433475, 'y': 364.941057936045, 'width': 278.90913365273093, 'height': 209.8713396547812},
		},
	}


	

	@classmethod
	def replacement_paths_dict(cls, active_figure_export_folder_date: str = '2026-11-04'):
		return {
			'Artboard 3.pdf': Path(f'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/2026-09-28 - EXPORTS FOR PUBLICATION/{active_figure_export_folder_date}/Figure 3 - FRI Scatter/_TEMP'),
			'Artboard 4.pdf': Path(f'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure 4 - P_Short Scatter over Time/{active_figure_export_folder_date}'),
			'figS3_TimeBinDurationOnDecodingPerform.pdf': Path(f'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure S3 - Effect of Time Bin Duration on Decoding Performance/Inputs/{active_figure_export_folder_date}'),
			'Figure S8 - Effect of Heuristic-filtering perTimeBin.pdf': Path(f'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure S8 - Effect of Heuristic-filtering P_Short Scatter over Time - perEpoch/Inputs/{active_figure_export_folder_date}'),
			# 'Figure S9': Path(f'E:/Dropbox (Personal)/Active/Kamran Diba Lab/Pho-Kamran-Meetings/Pho Diba Paper 2023/Figures/UneditedSources/Figure S9 - Effect of Heuristic-filtering P_Short Scatter over Time - PerEpoch/Inputs/{active_figure_export_folder_date}'),
		}


	@classmethod
	def replace_with_updated_images(cls, active_figure_name: str, **kwargs) -> Any:
		from vectorcraft_scripting import connect_first, replace_layers

		active_figure_export_folder_date: str = kwargs.pop('active_figure_export_folder_date', datetime.now().strftime('%Y-%m-%d'))
		live = kwargs.pop('live', None)
		backup_layers: bool = kwargs.pop('backup_layers', False)
		print_backup_paths: bool = kwargs.pop('print_backup_paths', True)
		if (live is None) or (not live.is_connected):
			live = connect_first()

		replacement_paths_dict = cls.replacement_paths_dict(active_figure_export_folder_date=active_figure_export_folder_date)
		curr_fig_geom_dict = cls.figs_layer_geometry[active_figure_name]
		curr_replacement_search_path: Path = replacement_paths_dict[active_figure_name]

		live.select_document(active_figure_name)  # activate the matching open tab
		live.assert_active_document(active_figure_name)  # fail before mutating if the wrong tab is active

		backup_paths = live.backup_document(expect_title=active_figure_name)  # e.g. {'same_format': .../_BAK/Artboard 4_2026-10-29.pdf, 'vectorcraft': .../_BAK/Artboard 4_2026-10-29.vectorcraft}
		if print_backup_paths:
			print(f'backup_paths: {backup_paths}')

		undo_day_date: str = datetime.now().strftime('%Y-%m-%d')
		with live.undo_group(f'Replace figure layers ({undo_day_date})'):
			replacement_df = replace_layers(live, curr_fig_geom_dict, curr_replacement_search_path, expect_title=active_figure_name, backup_layers=backup_layers, **kwargs)

		return replacement_df


# # %%
# outputs_folder: Path = _notebook_dir.joinpath('outputs').resolve()
# outputs_folder.mkdir(parents=True, exist_ok=True)
# print(f'outputs_folder: {outputs_folder}')
# print(f'file_uri: {file_uri_from_path(outputs_folder)}')

# # %%
# input_data_folder: Path = _notebook_dir.parent.joinpath('data').resolve()
# input_data_folder.mkdir(parents=False, exist_ok=True)
# print(f'input_data_folder: {input_data_folder}')
# print(f'file_uri: {file_uri_from_path(input_data_folder)}')

