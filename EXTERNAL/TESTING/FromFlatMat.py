# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:27:30 2021

@author: Pho
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# NeuroPy (Diba Lab Python Repo) Loading
from neuropy import core
from neuropy.core.dataSession import DataSessionLoader, DataSession, processDataSession
from neuropy.core.neurons import FlattenedSpiketrains, Neurons
from neuropy.core.position import Position
from neuropy.utils.load_exported import import_mat_file

## 08:
basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'
spike_file = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.spikeII.mat'
neuroscope_xml_file = Path(basedir).joinpath('2006-6-08_14-26-15.xml')

basedir = Path(basedir)
session_name = basedir.parts[-1]

## Load session:
curr_args_dict = dict()
curr_args_dict['basepath'] = basedir
curr_args_dict['session_obj'] = DataSession()
# curr_session = DataSessionLoader.default_load_bapun_npy_session_folder(curr_args_dict)
sess = DataSessionLoader.default_load_kamran_flat_spikes_mat_session_folder(curr_args_dict)
sess


position_mat_file_path = Path(basedir).joinpath('{}vt.mat'.format(session_name))
position_mat_file = import_mat_file(mat_import_file=position_mat_file_path)
tt = position_mat_file['tt'] # 1, 63192
xx = position_mat_file['xx'] # 10 x 63192
yy = position_mat_file['yy'] # 10 x 63192
tt = tt.flatten()
tt_rel = tt - tt[0] # relative timestamps
timestamps_conversion_factor = 1e6
#timestamps_conversion_factor = 1.0
t = tt / timestamps_conversion_factor  # (63192,)
t_rel = tt_rel / timestamps_conversion_factor  # (63192,)
position_sampling_rate_Hz = 1.0 / np.mean(np.diff(tt / 1e6)) # In Hz, returns 29.969777
num_samples = len(t);
x = xx[0,:].flatten() # (63192,)
y = yy[0,:].flatten() # (63192,)
active_t_start = t[0] # absolute t_start
#active_t_start = 0.0 # relative t_start
curr_pos_obj = Position.from_vt_mat_file(position_mat_file_path = Path(basedir).joinpath('{}vt.mat'.format(session_name)))

print('tt[0]: {}\n tt[-1]: {}'.format(tt[0], tt[-1]))
print('tt_rel[0]: {}\n tt_rel[-1]: {}'.format(tt_rel[0], tt_rel[-1]))

tt[-1]:
  
605893940268.0
68624338.0 * 1e4 # not sure why 1e4, but this value converts the times into

605893940268.0
686243380000.0


print('curr_pos_obj.time: {}\n curr_pos_obj.duration: {}\n curr_pos_obj.time[-1]: {}\n curr_pos_obj.time[0]: {}\n curr_pos_obj.sampling_rate: {}\n curr_pos_obj.t_start: {}\n curr_pos_obj.t_stop: {}\n'.format(
    curr_pos_obj.time,
    curr_pos_obj.duration,
    curr_pos_obj.time[-1],
    curr_pos_obj.time[0],
    curr_pos_obj.sampling_rate,
    curr_pos_obj.t_start,
    curr_pos_obj.t_stop))



# print("type is: ",type(flat_spikes_data)) # type is:  <class 'numpy.ndarray'>
# print("dtype is: ", flat_spikes_data.dtype) # dtype is:  [('t', 'O'), ('shank', 'O'), ('cluster', 'O'), ('aclu', 'O'), ('qclu', 'O'), ('cluinfo', 'O'), ('x', 'O'), ('y', 'O'), ('speed', 'O'), ('traj', 'O'), ('lap', 'O'), ('gamma2', 'O'), ('amp2', 'O'), ('ph', 'O'), ('amp', 'O'), ('gamma', 'O'), ('gammaS', 'O'), ('gammaM', 'O'), ('gammaE', 'O'), ('gamma2S', 'O'), ('gamma2M', 'O'), ('gamma2E', 'O'), ('theta', 'O'), ('ripple', 'O')]
mat_variables_to_extract = ['t', 'shank', 'cluster', 'aclu', 'qclu', 'cluinfo','x','y','speed','traj','lap']
num_mat_variables = len(mat_variables_to_extract)
flat_spikes_out_dict = dict()
for i in np.arange(num_mat_variables):
    curr_var_name = mat_variables_to_extract[i]
    if curr_var_name == 'cluinfo':
        temp = flat_spikes_data[curr_var_name][0,0] # a Nx4 array
        temp = [tuple(temp[j,:]) for j in np.arange(np.shape(temp)[0])]
        flat_spikes_out_dict[curr_var_name] = temp
    else:
        flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name][0,0].flatten()
spikes_df = pd.DataFrame(flat_spikes_out_dict) # 1014937 rows × 11 columns

def import_data_dir_with_flattened_spikes_mats(basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'):
    from neuropy.core.neurons import FlattenedSpiketrains, Neurons
    print('import_data_dir_with_flattened_spikes_mats(...):')
    basedir = Path(basedir)
    session_name = basedir.parts[-1]
    print('\t basedir: {}\n\t session_name: {}'.format(basedir, session_name)) # session_name: 2006-6-08_14-26-15
    neuroscope_xml_file = Path(basedir).joinpath('{}.xml'.format(session_name))
    spike_mat_file = Path(basedir).joinpath('{}.spikeII.mat'.format(session_name))
    print('\t neuroscope_xml_file: {}\n\t spike_mat_file: {}\n'.format(neuroscope_xml_file, spike_mat_file)) # session_name: 2006-6-08_14-26-15
    
    position_mat_file = Path(basedir).joinpath('{}vt.mat'.format(session_name))
    
    
    tempSession = NeuroscopeIO(neuroscope_xml_file)
    # tempSession.dat_sampling_rate # 32552
    # tempSession.eeg_sampling_rate # 1252
    # tempSession.n_channels # 96
    # tempSession.discarded_channels # array([], dtype=int32)
    # tempSession.skipped_channels # array([], dtype=float64)
    ## Matlab (.mat) spikes import:
    flat_spikes_mat_file = import_mat_file(mat_import_file=spike_mat_file)
    # print('flat_spikes_mat_file.keys(): {}'.format(flat_spikes_mat_file.keys())) # flat_spikes_mat_file.keys(): dict_keys(['__header__', '__version__', '__globals__', 'spike'])
    flat_spikes_data = flat_spikes_mat_file['spike']
    # print("type is: ",type(flat_spikes_data)) # type is:  <class 'numpy.ndarray'>
    # print("dtype is: ", flat_spikes_data.dtype) # dtype is:  [('t', 'O'), ('shank', 'O'), ('cluster', 'O'), ('aclu', 'O'), ('qclu', 'O'), ('cluinfo', 'O'), ('x', 'O'), ('y', 'O'), ('speed', 'O'), ('traj', 'O'), ('lap', 'O'), ('gamma2', 'O'), ('amp2', 'O'), ('ph', 'O'), ('amp', 'O'), ('gamma', 'O'), ('gammaS', 'O'), ('gammaM', 'O'), ('gammaE', 'O'), ('gamma2S', 'O'), ('gamma2M', 'O'), ('gamma2E', 'O'), ('theta', 'O'), ('ripple', 'O')]
    mat_variables_to_extract = ['t', 'shank', 'cluster', 'aclu', 'qclu', 'cluinfo','x','y','speed','traj','lap']
    num_mat_variables = len(mat_variables_to_extract)
    flat_spikes_out_dict = dict()
    for i in np.arange(num_mat_variables):
        curr_var_name = mat_variables_to_extract[i]
        if curr_var_name == 'cluinfo':
            temp = flat_spikes_data[curr_var_name][0,0] # a Nx4 array
            temp = [tuple(temp[j,:]) for j in np.arange(np.shape(temp)[0])]
            flat_spikes_out_dict[curr_var_name] = temp
        else:
            flat_spikes_out_dict[curr_var_name] = flat_spikes_data[curr_var_name][0,0].flatten()
    spikes_df = pd.DataFrame(flat_spikes_out_dict) # 1014937 rows × 11 columns
    classNames = ['pyramidal','contaminated','interneurons']
    classCutoffValues = [0, 4, 7, 9]
    spikes_df['neuron_type'] = pd.cut(x=spikes_df['qclu'], bins=classCutoffValues, labels=classNames)
    # unique_cell_ids, unique_cell_id_indices = np.unique(flat_spikes_out_dict['aclu'], return_index=True)
    unique_cell_ids = np.unique(flat_spikes_out_dict['aclu'])
    flat_cell_ids = [int(cell_id) for cell_id in unique_cell_ids] 
    # print('flat_cell_ids: {}'.format(flat_cell_ids))
    # Group by the aclu (cluster indicator) column
    cell_grouped_spikes_df = spikes_df.groupby(['aclu'])
    num_unique_cell_ids = len(flat_cell_ids)
    spiketrains = list()
    shank_ids = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
    cell_quality = np.zeros([num_unique_cell_ids, ]) # (108,) Array of float64
    neuron_type = list() # (108,) Array of float64
    
    for i in np.arange(num_unique_cell_ids):
        curr_cell_id = flat_cell_ids[i] # actual cell ID
        #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
        curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
        spiketrains.append(curr_cell_dataframe['t'].to_numpy())
        shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
        cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
        neuron_type.append(curr_cell_dataframe['neuron_type'].to_numpy()[0])
        
    spiketrains = np.array(spiketrains, dtype='object')
    t_stop = np.max(flat_spikes_out_dict['t'])
    # print('t_stop: {}'.format(t_stop))
    # flattened_spike_times = flat_spikes_out_dict['t']
    # flattened_spike_identities = flat_spikes_out_dict['aclu']
    # # Get the indicies required to sort the flattened_spike_times
    # sorted_indicies = np.argsort(flattened_spike_times)
    # flattened_spikeTrainsObj = FlattenedSpiketrains(sorted_indicies, flattened_spike_identities, flattened_spike_times)
    # flattened_spikeTrainsObj
    
    
    curr_Neurons_obj = Neurons(spiketrains, t_stop, t_start=0,
            sampling_rate=tempSession.dat_sampling_rate,
            neuron_ids=flat_cell_ids,
            neuron_type=neuron_type,
            shank_ids=shank_ids
    )
    position_obj = Position.from_separate_arrays(flat_spikes_out_dict['t'], flat_spikes_out_dict['x'], flat_spikes_out_dict['y'])
    print('done.')
    return curr_Neurons_obj, position_obj
    
neurons_obj, position_obj = import_data_dir_with_flattened_spikes_mats(basedir)

np.isnan

print (spikes_df.x.first_valid_index()) # 20
print (spikes_df.x.loc[spikes_df.x.first_valid_index()]) #0.8059081439677763

print (spikes_df.y.first_valid_index()) # 20
print (spikes_df.y.loc[spikes_df.y.first_valid_index()]) #0.4936032096919495

print (spikes_df.x.last_valid_index()) # 1006874
print (spikes_df.x.loc[spikes_df.x.last_valid_index()]) #0.7592097744851032

print (spikes_df.y.last_valid_index()) # 1006874
print (spikes_df.y.loc[spikes_df.y.last_valid_index()]) #0.49604708929347086


# Range of the maze epoch (where position is valid):

t_maze_start = spikes_df.t.loc[spikes_df.x.first_valid_index()] # 1048
t_maze_end = spikes_df.t.loc[spikes_df.x.last_valid_index()] # 68159707


spikes_df.t.min() # 88
spikes_df.t.max() # 68624338

#epochs_df = pd.Dataframe({'start':[0,2000,3000],'stop':[2000,3000,6000],'label':['pre','maze','post']})

epochs_df = pd.DataFrame({'start':[0, t_maze_start, t_maze_end],'stop':[t_maze_start, t_maze_end, spikes_df.t.max()],'label':['pre','maze','post']})
