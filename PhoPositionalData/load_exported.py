#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pho
"""
import sys
import numpy as np
import h5py
import hdf5storage # conda install hdf5storage
from pathlib import Path
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, print_file_progress_message

def import_mat_file(mat_import_file='data/RoyMaze1/positionAnalysis.mat'):
    with ProgressMessagePrinter(mat_import_file, 'Loading', 'matlab import file'):
        data = hdf5storage.loadmat(mat_import_file, appendmat=False)
    return data

# ## Load Spiking Information:
# # load_path = Path('/Volumes/iNeo/Data/Rotation_3_Kamran Diba Lab/ClusterFreeAnalysisProject/Data/Achilles_10252013/ExportedData/Achilles_10252013_Output_All.npz')
# load_path = Path('data/Achilles_10252013/ExportedData/Achilles_10252013_Output_All.npz')
# ## Load Previously Saved Data:
# active_filename = load_path
# with open(active_filename, 'rb') as f:
#     loaded_data = np.load(f)
#     loaded_files = loaded_data.files
#     data_timestamps = loaded_data['data_timestamps']
#     data_position = loaded_data['spike_position_rate']
#     data_spikes = loaded_data['data_spikes']
#     spike_position_rate = loaded_data['spike_position_rate']
#     print('loaded loaded_data from {}'.format(active_filename))
    
# np.shape(data_spikes)