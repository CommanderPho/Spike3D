import sys
from PhoNonInteractiveTest import PhoNonInteractiveTest
from neuropy.plotting.ratemaps import plot_ratemap_2D, enumTuningMap2DPlotVariables, enumTuningMap2DPlotMode


if __name__ == "__main__":    
    test = PhoNonInteractiveTest()
    active_epoch_placefields1D, active_epoch_placefields2D, even_lap_specific_placefields1D, even_lap_specific_placefields2D, odd_lap_specific_placefields1D, odd_lap_specific_placefields2D, any_lap_specific_placefields1D, any_lap_specific_placefields2D, active_config, active_epoch_session, good_placefields_session = PhoNonInteractiveTest.run(test.sess, test.active_sess_config)
    
    ## Cell:
    fig = plot_ratemap_2D(active_epoch_placefields2D.ratemap, figsize=None, fig_column_width=4.0, fig_row_height=1.0,  subplots=(None, 3), enable_spike_overlay=True, spike_overlay_spikes=active_epoch_placefields2D.spk_pos, included_unit_indicies=None)
    print(fig)
