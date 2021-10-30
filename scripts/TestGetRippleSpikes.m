% behavioral_duration_indicies
% behavioral_states
% behavioral_epoch

% For each ripple
active_ripples = source_data.ripple.RoyMaze1.time;
num_ripples = size(active_ripples, 1);

if ~exist('rippleSpikes','var')
    rippleSpikes = {};
    for ripple_idx = 1:num_ripples
        rippleSpikes{ripple_idx} = cellfun(@(curr_spikes) curr_spikes((active_ripples(ripple_idx,1) <= curr_spikes) & (active_ripples(ripple_idx,2) <= curr_spikes)), spike_cells, 'UniformOutput', false);
    end
end





rippleSpikeCounts = cellfun(@(currRippleSpikes) sum(cellfun(@(x) sum(x,"all"), currRippleSpikes),"all"), rippleSpikes);



