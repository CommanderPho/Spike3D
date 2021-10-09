% Saves out the behavioral_epochs table data and the behavioral_periods
% table for use in python.
if ~exist('timesteps_array','var')
    load('PhoResults_Expt1_RoyMaze1.mat', 'timesteps_array')
    load('PhoResults_Expt1_RoyMaze1.mat', 'active_processing')
end

out_filepath = 'C:\Share\data\RoyMaze1\ExportedData';

% behavioral_epochs = table2dataset(active_processing.behavioral_epochs);
% behavioral_periods = table2dataset(active_processing.behavioral_periods_table);
% 
% behavioral_epochs = table2array(active_processing.behavioral_epochs);
% behavioral_periods = table2array(active_processing.behavioral_periods_table);

% behavioral_epochs = table2cell(active_processing.behavioral_epochs);
% behavioral_periods = table2cell(active_processing.behavioral_periods_table);

% behavioral_periods.epoch_start_seconds = double(active_processing.behavioral_periods_table.epoch_start_seconds);
% behavioral_periods.epoch_end_seconds = double(active_processing.behavioral_periods_table.epoch_end_seconds);
% behavioral_periods.duration = double(active_processing.behavioral_periods_table.duration);
% behavioral_periods.behavioral_epoch = double(active_processing.behavioral_periods_table.behavioral_epoch);
% behavioral_periods.type = double(active_processing.behavioral_periods_table.type);

% Numerical table version:
behavioral_epochs = [[0:(height(active_processing.behavioral_epochs)-1)]', table2array(active_processing.behavioral_epochs)];
behavioral_periods = [[0:(height(active_processing.behavioral_periods_table)-1)]', double(active_processing.behavioral_periods_table.epoch_start_seconds), double(active_processing.behavioral_periods_table.epoch_end_seconds), double(active_processing.behavioral_periods_table.duration), double(active_processing.behavioral_periods_table.behavioral_epoch), double(active_processing.behavioral_periods_table.type)];

save(fullfile(out_filepath, 'extrasAnalysis.mat'), 'behavioral_epochs', 'behavioral_periods')

