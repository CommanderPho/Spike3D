import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from collections import defaultdict
from functools import partial
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pyphocorehelpers.assertion_helpers import Assert
from pyphocorehelpers.print_helpers import print_keys_if_possible
from pyphocorehelpers.Filesystem.path_helpers import find_first_extant_path, file_uri_from_path, sanitize_filename_for_Windows


from vectorcraft_scripting import (
    CliNotFoundError,
    PlaceSpec,
    VectorCraftClient,
    VectorCraftError,
    export_plotly_figure,
    layer_geometry,
    place_into_template,
    replace_layers,
    write_plotly_svg,
)
from vectorcraft_scripting import ObjectQuery, connect_first, discover_instances, NoInstanceError, LiveVectorCraft
from Spike3D.EXTERNAL.VectorCraft_Scripting.src.vectorcraft_scripting.scripts.__init__ import PhoDibaPaperVectorCraftHelper


active_figure_export_folder_date: str = '2026-11-04'
replacement_paths_dict = PhoDibaPaperVectorCraftHelper.replacement_paths_dict(active_figure_export_folder_date=active_figure_export_folder_date)


if ('live' not in globals()) or (live is None) or (not live.is_connected):
    live = connect_first()

## For a particular figure
# active_figure_name: str = 'Artboard 3.pdf'# 
# active_figure_name: str = 'Artboard 4.pdf'# 
active_figure_name: str = 'figS3_TimeBinDurationOnDecodingPerform.pdf'# 
# active_figure_name: str = 'Figure S8 - Effect of Heuristic-filtering perTimeBin.pdf'

## search for files with names matching those in figs_layer_geometry in `curr_replacement_search_path`
# for active_figure_name, curr_fig_geom_dict in figs_layer_geometry.items():
curr_fig_geom_dict = PhoDibaPaperVectorCraftHelper.figs_layer_geometry[active_figure_name]
curr_replacement_search_path: Path = replacement_paths_dict[active_figure_name]

live.select_document(active_figure_name)  # activate the matching open tab
live.assert_active_document(active_figure_name)  # fail before mutating if the wrong tab is active

backup_paths = live.backup_document(expect_title=active_figure_name)  # e.g. {'same_format': .../_BAK/Artboard 4_2026-10-29.pdf, 'vectorcraft': .../_BAK/Artboard 4_2026-10-29.vectorcraft}
print(f'backup_paths: {backup_paths}')
backup_layers_with_date: bool = False  # True: duplicate existing layers as hidden `<name>_<YYYY-MM-DD>` copies before placing
with live.undo_group(f'Replace figure layers ({datetime.now():%Y-%m-%d})'):
    replacement_df = replace_layers(live, curr_fig_geom_dict, curr_replacement_search_path, expect_title=active_figure_name, backup_layers=backup_layers_with_date)
replacement_df

