# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:27:30 2021

@author: Pho
"""


from neuropy.core.neurons import FlattenedSpiketrains, Neurons
from neuropy.core.position import Position


def import_data_dir_with_flattened_spikes_mats(basedir = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15'):
    from neuropy.core.neurons import FlattenedSpiketrains, Neurons
    print('import_data_dir_with_flattened_spikes_mats(...):')
    basedir = Path(basedir)
    session_name = basedir.parts[-1]
    print('\t basedir: {}\n\t session_name: {}'.format(basedir, session_name)) # session_name: 2006-6-08_14-26-15
    spike_file = r'R:\data\KDIBA\gor01\one\2006-6-08_14-26-15\2006-6-08_14-26-15.spikeII.mat'
    neuroscope_xml_file = Path(basedir).joinpath('{}.xml'.format(session_name))
    spike_mat_file = Path(basedir).joinpath('{}.spikeII.mat'.format(session_name))
    print('\t neuroscope_xml_file: {}\n\t spike_mat_file: {}\n'.format(neuroscope_xml_file, spike_mat_file)) # session_name: 2006-6-08_14-26-15
    tempSession = NeuroscopeIO(neuroscope_xml_file)
    # tempSession.dat_sampling_rate # 32552
    # tempSession.eeg_sampling_rate # 1252
    # tempSession.n_channels # 96
    # tempSession.discarded_channels # array([], dtype=int32)
    # tempSession.skipped_channels # array([], dtype=float64)
    ## Matlab (.mat) spikes import:
    flat_spikes_mat_import_file = spike_file
    flat_spikes_mat_file = import_mat_file(mat_import_file=flat_spikes_mat_import_file)
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
    spikes_df['cell_type'] = pd.cut(x=spikes_df['qclu'], bins=classCutoffValues, labels=classNames)
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
    cell_type = list() # (108,) Array of float64
    
    for i in np.arange(num_unique_cell_ids):
        curr_cell_id = flat_cell_ids[i] # actual cell ID
        #curr_flat_cell_indicies = (flat_spikes_out_dict['aclu'] == curr_cell_id) # the indicies where the cell_id matches the current one
        curr_cell_dataframe = cell_grouped_spikes_df.get_group(curr_cell_id)
        spiketrains.append(curr_cell_dataframe['t'].to_numpy())
        shank_ids[i] = curr_cell_dataframe['shank'].to_numpy()[0] # get the first shank identifier, which should be the same for all of this curr_cell_id
        cell_quality[i] = curr_cell_dataframe['qclu'].mean() # should be the same for all instances of curr_cell_id, but use mean just to make sure
        cell_type.append(curr_cell_dataframe['cell_type'].to_numpy()[0])
        
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
            neuron_type=cell_type,
            shank_ids=shank_ids
    )
    temp_position_traces = np.vstack((flat_spikes_out_dict['t'], flat_spikes_out_dict['x'], flat_spikes_out_dict['y'])) # (3 x Nf)
    position_obj = Position(traces=temp_position_traces)
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

t_start = spikes_df.t.loc[spikes_df.x.first_valid_index()] # 1048
t_end = spikes_df.t.loc[spikes_df.x.last_valid_index()] # 68159707


spikes_df.t.min() # 88
spikes_df.t.max() # 68624338

