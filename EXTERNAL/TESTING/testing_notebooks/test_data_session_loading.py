# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import importlib
from threading import Thread
import time # for time.sleep
import numpy as np
import h5py
import hdf5storage # conda install hdf5storage
from pathlib import Path
from neuropy import core
from neuropy.core.session.data_session_loader import DataSessionLoader
# from neuropy.core.flattened_spiketrains import FlattenedSpiketrains
# from neuropy.core.position import Position
# from neuropy.core.session.KnownDataSessionTypeProperties import KnownDataSessionTypeProperties
# from neuropy.core.session.dataSession import DataSession
# from neuropy.core.session.Formats.SessionSpecifications import SessionFolderSpec, SessionFileSpec, SessionConfig, ParametersContainer
# from neuropy.core import SessionFolderSpec, DataSessionLoader, DataSession, processDataSession

# %%
# import PhoPositionalData as pdp
# from pyphoplacecellanalysis.PhoPositionalData. import load_exported, process_data
# import PhoPositionalData
# import PhoGui

# sys.path
# sys.cwd
# Path.cwd()
# Path.cd('PhoPy3DPositionAnalysis2021')
# import PhoPy3DPositionAnalysis2021.PhoPositionalData as phoPD

from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import *


# from phoPD.load_exported import *
# from pyphoplacecellanalysis.PhoPositionalData.process_data import process_positionalAnalysis_data, gen_2d_histrogram, get_heatmap_color_vectors, process_chunk_equal_poritions_data, extract_spike_timeseries
# from pyphoplacecellanalysis.PhoPositionalData.process_data import *
# from pyphoplacecellanalysis.PhoPositionalData.plot_data import *
# from pyphoplacecellanalysis.PhoPositionalData.plotting.animations import * # make_mp4_from_plotter
# from pyphoplacecellanalysis.PhoPositionalData.import_data import * # build_spike_positions_list, build_cellID_reverse_lookup_map
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import InteractivePlaceCellConfig, VideoOutputModeConfig, PlottingConfig, PlacefieldComputationParameters, NamedTimerange, SessionConfig  # VideoOutputModeConfig, InteractivePlaceCellConfigs
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import get_subsession_for_epoch, print_subsession_neuron_differences

# from pyphoplacecellanalysis.PhoPositionalData.load_exported import *
# # from pyphoplacecellanalysis.PhoPositionalData.process_data import process_positionalAnalysis_data, gen_2d_histrogram, get_heatmap_color_vectors, process_chunk_equal_poritions_data, extract_spike_timeseries
# from pyphoplacecellanalysis.PhoPositionalData.process_data import *
# from pyphoplacecellanalysis.PhoPositionalData.plot_data import *
# from pyphoplacecellanalysis.PhoPositionalData.plotting.animations import * # make_mp4_from_plotter
# from pyphoplacecellanalysis.PhoPositionalData.import_data import * # build_spike_positions_list, build_cellID_reverse_lookup_map
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import InteractivePlaceCellConfig, VideoOutputModeConfig, PlottingConfig, PlacefieldComputationParameters, NamedTimerange, SessionConfig  # VideoOutputModeConfig, InteractivePlaceCellConfigs
# from pyphoplacecellanalysis.PhoPositionalData.analysis.interactive_placeCell_config import get_subsession_for_epoch, print_subsession_neuron_differences


## Data must be pre-processed using the MATLAB script located here: 
# R:\data\KDIBA\gor01\one\IIDataMat_Export_ToPython_2021_11_23.m

# # From pre-computed .mat files:
# ## 07: 
# basedir = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53'
# # spike_file = r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53\2006-6-07_11-26-53.spikeII.mat'
# # neuroscope_xml_file = Path(basedir).joinpath('2006-6-07_11-26-53.xml')

# # ## 08:
# # basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'
# # spike_file = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.spikeII.mat' # '2006-6-08_14-26-15.spikeII.mat' # Contains 'spike' flat structure
# # neuroscope_xml_file = Path(basedir).joinpath('2006-6-08_14-26-15.xml')

# # sess = core.processDataSession(basedir)
# session_name = Path(basedir).parts[-1]

# # %% [markdown]
# # ## KDiba Old Format:

# # %%
# # KDiba Old Format:
# sess = DataSessionLoader.kdiba_old_format_session(r'R:\data\KDIBA\gor01\one\2006-6-07_11-26-53')
# sess.is_resolved

# %% [markdown]
# ## Bapun Format:

# %%
## Bapun Format:
# basedir = '/media/share/data/Bapun/Day5TwoNovel' # Linux
# basedir = Path('R:\data\Bapun\Day5TwoNovel') # Windows
basedir = Path(r'M:\Data\Bapun\RatS\Day5TwoNovel') # Windows
# basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel' # MacOS
sess = DataSessionLoader.bapun_data_session(basedir)


print(f'done!')

# %%
sess.is_resolved

# %% [markdown]
# ## Other:

# %%



