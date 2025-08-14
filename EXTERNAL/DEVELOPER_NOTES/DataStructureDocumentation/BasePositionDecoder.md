BasePositionDecoder - printed by print_keys_if_possible on 2025-01-27
===================================================================================================


    BasePositionDecoder: pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BasePositionDecoder  = BasePositionDecoder(pf=PfND(spikes_df=                 t      t_seconds  t_rel_seconds  shank  cluster  aclu  qclu           x           y      speed  traj  lap  maze_relative_lap  maze_id  is_theta  is_ripple  theta_phase_radians           neuron_type  flat_spike_idx  x_loade...
	│   ├── pf: neuropy.analyses.placefields.PfND  = PfND(spikes_df=                 t      t_seconds  t_rel_seconds  shank  cluster  aclu  qclu           x           y      speed  traj  lap  maze_relative_lap  maze_id  is_theta  is_ripple  theta_phase_radians           neuron_type  flat_spike_idx  x_loaded  y_loaded     lin_pos...
		│   ├── spikes_df: pandas.core.frame.DataFrame (children omitted) - (87501, 25)
		│   ├── position: neuropy.core.position.Position  = <Position: {'_filename': None, '_metadata': None, '_data':                  t     lin_pos           x        dt  velocity_x  acceleration_x    x_smooth  velocity_x_smooth  acceleration_x_smooth      speed<br>0         0.150794  155.019823  155.019823  0.000000    0.000000     ...
			│   ├── _filename: NoneType  = None
			│   ├── _metadata: NoneType  = None
			│   ├── _data: pandas.core.frame.DataFrame (children omitted) - (30844, 10)
		│   ├── epochs: neuropy.core.epoch.Epoch  = 22 epochs<br>array([[3.05477, 4.72322],<br>       [45.196, 50.87],<br>       [84.3034, 91.0442],<br>       [244.13, 250.671],<br>       [281.302, 290.778],<br>       [357.845, 365.118],<br>       [399.654, 406.36],<br>       [428.285, 435.323],<br>       [462.884, 470.192],<br...
			│   ├── _filename: NoneType  = None
			│   ├── _metadata: NoneType  = None
			│   ├── _df: pandas.core.frame.DataFrame (children omitted) - (22, 6)
		│   ├── config: neuropy.analyses.placefields.PlacefieldComputationParameters  = <PlacefieldComputationParameters: {'speed_thresh': 10.0, 'grid_bin': (3.8054171165052444, 1.4477079927649104), 'grid_bin_bounds': ((0.0, 287.7697841726619), (115.10791366906477, 172.66187050359713)), 'smooth': (2.0, 2.0), 'frate_thresh': 1.0, 'is_directional': False};>
			│   ├── speed_thresh: float  = 10.0
			│   ├── grid_bin: tuple  = (3.8054171165052444, 1.4477079927649104) - (2,)
			│   ├── grid_bin_bounds: tuple  = ((0.0, 287.7697841726619), (115.10791366906477, 172.66187050359713)) - (2, 2)
			│   ├── smooth: tuple  = (2.0, 2.0) - (2,)
			│   ├── frate_thresh: float  = 1.0
			│   ├── is_directional: bool  = False
		│   ├── position_srate: numpy.float64  = 29.969756198785802
		│   ├── setup_on_init: bool  = True
		│   ├── compute_on_init: bool  = True
		│   ├── _save_intermediate_spikes_maps: bool  = True
		│   ├── _ratemap: neuropy.core.ratemap.Ratemap  = <Ratemap: {'_filename': None, '_metadata': None, 'spikes_maps': array([[0, 0, 0, ..., 0, 0, 0],<br>       [0, 0, 0, ..., 0, 0, 0],<br>       [0, 0, 0, ..., 0, 0, 0],<br>       ...,<br>       [0, 0, 0, ..., 0, 0, 0],<br>       [0, 0, 0, ..., 0, 0, 0],<br>       [0, 0, 0, ..., 0...
			│   ├── _filename: NoneType  = None
			│   ├── _metadata: NoneType  = None
			│   ├── spikes_maps: numpy.ndarray  = [[0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]<br> ...<br> [0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]] - (35, 76)
			│   ├── tuning_curves: numpy.ndarray  = [[0 3.39909e-05 0.000221649 ... 0 0 0]<br> [0 0 2.21598e-05 ... 0 0 0]<br> [0 0.000203946 0.00149609 ... 0.00130773 0.000200546 0]<br> ...<br> [0 1.13303e-05 7.38829e-05 ... 0.000653864 0.000100273 0]<br> [0 0 1.66199e-05 ... 0 0 0]<br> [0 6.79819e-05 0.000476537 ... 0.0006538... - (35, 76)
			│   ├── unsmoothed_tuning_maps: numpy.ndarray  = [[0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]<br> ...<br> [0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]<br> [0 0 0 ... 0 0 0]] - (35, 76)
			│   ├── _neuron_ids: list  = [4, 9, 11, 12, 16, 18, 20, 24, 27, 28, 31, 38, 39, 44, 48, 57, 58, 59, 61, 63, 66, 67, 72, 79, 80, 81, 82, 84, 89, 90, 91, 92, 93, 98, 104] - (35,)
			│   ├── _neuron_extended_ids: list  = [NeuronExtendedIdentity(shank=1, cluster=4, aclu=4, qclu=2), NeuronExtendedIdentity(shank=1, cluster=14, aclu=9, qclu=2), NeuronExtendedIdentity(shank=1, cluster=17, aclu=11, qclu=2), NeuronExtendedIdentity(shank=1, cluster=18, aclu=12, qclu=1), NeuronExtendedIdentity(shank=2,... - (35,)
			│   ├── xbin: numpy.ndarray  = [0 3.80542 7.61083 11.4163 15.2217 19.0271 22.8325 26.6379 30.4433 34.2488 38.0542 41.8596 45.665 49.4704 53.2758 57.0813 60.8867 64.6921 68.4975 72.3029 76.1083 79.9138 83.7192 87.5246 91.33 95.1354 98.9408 102.746 106.552 110.357 114.163 117.968 121.773 125.579 129.384 133.1... - (77,)
			│   ├── ybin: NoneType  = None
			│   ├── occupancy: numpy.ndarray  = [0 0 0 0 0 0 0 0 0 5.90595 12.0788 5.33872 2.30232 3.03639 4.40444 0 0 0 2.23559 0.800807 3.1365 2.56926 1.56825 2.26895 2.13549 2.40242 0.900908 1.80182 1.33468 1.10111 1.26794 1.26794 1.06774 1.20121 1.10111 1.20121 1.23458 1.83518 2.10212 1.56825 1.33468 1.50151 1.43478 1.3... - (76,)
		│   ├── _ratemap_spiketrains: list  = [array([45.7893, 46.8701, 47.0713, 50.3183, 50.6422, 87.849, 87.8549, 87.8873, 87.8917, 87.9411, 247.518, 247.796, 287.445, 287.548, 287.552, 290.245, 362.2, 362.431, 432.303, 432.731, 467.023, 467.173, 467.193, 467.297, 467.301, 467.352, 467.379, 467.385, 493.542, 493.848, 51... - (35,)
		│   ├── _ratemap_spiketrains_pos: list  = [[array([214.807, 162.359, 150.904, 36.5565, 37.0682, 156.341, 156.084, 154.712, 154.535, 152.455, 148.282, 135.478, 159.97, 155.045, 154.857, 35.827, 152.825, 141.027, 163.85, 142.41, 160.031, 152.247, 151.172, 145.509, 145.282, 142.374, 141.034, 140.716, 155.442, 134.265, 14... - (35, 1)
		│   ├── _filtered_pos_df: pandas.core.frame.DataFrame (children omitted) - (4171, 11)
		│   ├── _filtered_spikes_df: pandas.core.frame.DataFrame (children omitted) - (9340, 25)
		│   ├── ndim: numpy.int32  = 1
		│   ├── xbin: numpy.ndarray  = [0 3.80542 7.61083 11.4163 15.2217 19.0271 22.8325 26.6379 30.4433 34.2488 38.0542 41.8596 45.665 49.4704 53.2758 57.0813 60.8867 64.6921 68.4975 72.3029 76.1083 79.9138 83.7192 87.5246 91.33 95.1354 98.9408 102.746 106.552 110.357 114.163 117.968 121.773 125.579 129.384 133.1... - (77,)
		│   ├── ybin: NoneType  = None
		│   ├── bin_info: dict (children omitted)(all scalar values) - size: 3
		│   ├── _included_thresh_neurons_indx: NoneType  = None
		│   ├── _peak_frate_filter_function: NoneType  = None
	│   ├── neuron_IDXs: numpy.ndarray  = [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34] - (35,)
	│   ├── neuron_IDs: numpy.ndarray  = [  4   9  11  12  16  18  20  24  27  28  31  38  39  44  48  57  58  59  61  63  66  67  72  79  80  81  82  84  89  90  91  92  93  98 104] - (35,)
	│   ├── F: numpy.ndarray  = [[0 0 0 ... 0 0 0]<br> [3.39909e-05 0 0.000203946 ... 1.13303e-05 0 6.79819e-05]<br> [0.000221649 2.21598e-05 0.00149609 ... 7.38829e-05 1.66199e-05 0.000476537]<br> ...<br> [0 0 0.00130773 ... 0.000653864 0 0.000653864]<br> [0 0 0.000200546 ... 0.000100273 0 0.000100273]<br> ... - (76, 35)
	│   ├── P_x: numpy.ndarray  = [[0]<br> [0]<br> [0]<br> [0]<br> [0]<br> [0]<br> [0]<br> [0]<br> [0]<br> [5.90595]<br> [12.0788]<br> [5.33872]<br> [2.30232]<br> [3.03639]<br> [4.40444]<br> [0]<br> [0]<br> [0]<br> [2.23559]<br> [0.800807]<br> [3.1365]<br> [2.56926]<br> [1.56825]<br> [2.26895]<br> [2.13549]<br... - (76, 1)
	│   ├── setup_on_init: bool  = True
	│   ├── post_load_on_init: bool  = False
	│   ├── debug_print: bool  = False
