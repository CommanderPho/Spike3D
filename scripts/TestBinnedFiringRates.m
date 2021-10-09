load('spikesAnalysis.mat') % for spike_positions_list
load('PhoResults_Expt1_RoyMaze1.mat', 'timesteps_array')
load('PhoResults_Expt1_RoyMaze1.mat', 'active_processing')

% The binned spike counts for each cell
% 
%active_processing.processed_array{1, 1}.all.binned_spike_counts

[is_period_track_active] = fnFilterPeriodsWithCriteria(active_processing, {'track'}, {'active'}); % Only get the active periods on the track


binned_spike_count_matrix = cell2mat(active_processing.processed_array{1, 1}.all.binned_spike_counts); % NumTimestamps x NumCells

total_spike_counts = sum(binned_spike_count_matrix, 1);
mean_spike_counts = mean(binned_spike_count_matrix, 1);
max_spike_counts = max(binned_spike_count_matrix, [], 1);
spike_counts_variance = var(binned_spike_count_matrix, 1);
% Build a table for results:
spike_counts_info_table = table([1:length(total_spike_counts)]', total_spike_counts', mean_spike_counts', max_spike_counts', spike_counts_variance', active_processing.spikes.isAlwaysStable, ...
    'VariableNames',{'cell_id', 'total_spike_counts', 'mean_spike_counts', 'max_spike_counts', 'spike_counts_variance', 'isAlwaysStable'});

load('C:\Users\Pho\repos\PhoPy3DPositionAnalysis2021\output\all_spike_list_testing.mat')
num_cells = length(spike_positions_list); % 92

num_spatial_bins = 80;
smoothing_sigma = 4;

spatial_histogram = {};
spatial_heatmap = {};
spatial_heatmap_smoothed = {};
spatial_selectivity = zeros([num_cells, 1]);

for curr_cell_index = 1:num_cells
    curr_cell_spikes = (spike_positions_list{curr_cell_index});
    spatial_histogram{curr_cell_index} = histogram2(curr_cell_spikes(1,:), curr_cell_spikes(2,:), num_spatial_bins, 'Normalization', 'countdensity');
    % spatial_histogram = gaussian_filter(spatial_histogram, sigma=smoothing_sigma);
    spatial_heatmap{curr_cell_index} = spatial_histogram{curr_cell_index}.BinCounts;
    spatial_heatmap_smoothed{curr_cell_index} = smoothdata(spatial_heatmap{curr_cell_index},'gaussian', smoothing_sigma);
    
    % spatial_selectivity = sum(nonzeros(spatial_heatmap), 'all')
    spatial_selectivity(curr_cell_index) = nnz(spatial_heatmap{curr_cell_index}); % nnz gets the count of the non-zero elements

end

% These are the same for all of them, so just get the last one
xedges = spatial_histogram{curr_cell_index}.XBinEdges;
yedges = spatial_histogram{curr_cell_index}.YBinEdges;
extent = [xedges(1), xedges(end), yedges(1), yedges(end)];

