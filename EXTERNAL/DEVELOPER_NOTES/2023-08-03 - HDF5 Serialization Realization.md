session_batch_status


I made an `HDF_Converter` class to manage how different types are serialized to HDF, like `pathlib.Path: str(v)`. For use as serialization_fn in `serialized_attribute_field(init=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v.to_dict())))`

Currently adding the outputs of the completion function which are currently statically defined `dict` and end up stored in `BatchRun.session_batch_outputs`

Going to make a separate serializable class to hold the output objects
Going to make a PyTables table defn from
```python
		session_batch_status: dict = Factory(dict)
		session_batch_basedirs: dict = Factory(dict)
		session_batch_errors: dict = Factory(dict)
		session_batch_outputs: dict = Factory(dict)

```
Plus:
```python
session_contexts = list(self.session_batch_status.keys())
```
and write it out in `to_hdf()` which will allow serializing the batch run's results in addition to the individual session outputs.
[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/runBatch.py:427](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/runBatch.py:427)




[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/runBatch.py:495](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/runBatch.py:495)
```python
@classmethod
    def build_batch_lap_replay_counts_df(cls, global_batch_run: BatchRun):
        """ Adds detected laps/replays to the batch_progress_df. returns lap_replay_counts_df """
        out_counts = []
        out_new_column_names = ['n_long_laps', 'n_long_replays', 'n_short_laps', 'n_short_replays']
        for ctx, output_v in global_batch_run.session_batch_outputs.items():
            if output_v is not None:
                # {long_epoch_name:(long_laps, long_replays), short_epoch_name:(short_laps, short_replays)}
                (long_laps, long_replays), (short_laps, short_replays) = list(output_v.values())[:2] # only get the first four outputs
                out_counts.append((long_laps.n_epochs, long_replays.n_epochs, short_laps.n_epochs, short_replays.n_epochs))
            else:
                out_counts.append((0, 0, 0, 0))
        return pd.DataFrame.from_records(out_counts, columns=out_new_column_names)
                


```

# IMPORTANT

### HDF_SerializationMixin, AttrsBasedClassHelperMixin, etc:
Conceptually, the hdf5 approach I made is about how objects are serialized out to file.
	My class definitions have different size members variables that need to be serialized. 
		- Some lists of dataframes, some nd.arrays of different shapes, etc.

```python
@custom_define(slots=False)
class InstantaneousSpikeRateGroupsComputation(HDF_SerializationMixin, AttrsBasedClassHelperMixin):
    """ class to handle spike rate computations 

    from pyphoplacecellanalysis.General.Batch.PhoDiba2023Paper import InstantaneousSpikeRateGroupsComputation

    """
    instantaneous_time_bin_size_seconds: float = serialized_attribute_field(default=0.01) # 20ms
    active_identifying_session_ctx: IdentifyingContext = serialized_attribute_field(init=False, serialization_fn=(lambda f, k, v: HDF_Converter._convert_dict_to_hdf_attrs_fn(f, k, v.to_dict()))) # need to write custom serialization to attributes I think

    LxC_aclus: np.ndarray = serialized_field(init=False, hdf_metadata={'track_eXclusive_cells': 'LxC'}) # the list of long-eXclusive cell aclus
    SxC_aclus: np.ndarray = serialized_field(init=False, hdf_metadata={'track_eXclusive_cells': 'SxC'}) # the list of short-eXclusive cell aclus
    
    Fig2_Replay_FR: List[SingleBarResult] = serialized_field(init=False, is_computable=True, serialization_fn=_InstantaneousSpikeRateGroupsComputation_convert_Fig2_ANY_FR_to_hdf_fn, hdf_metadata={'epochs': 'Replay'}) # a list of the four single-bar results.
    Fig2_Laps_FR: List[SingleBarResult] = serialized_field(init=False, is_computable=True, serialization_fn=_InstantaneousSpikeRateGroupsComputation_convert_Fig2_ANY_FR_to_hdf_fn, hdf_metadata={'epochs': 'Laps'}) # a list of the four single-bar results.

    LxC_ReplayDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaMinus'})
    LxC_ReplayDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaPlus'})
    SxC_ReplayDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaMinus'})
    SxC_ReplayDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Replay', 'track_change_relative_period': 'DeltaPlus'})

    LxC_ThetaDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaMinus'})
    LxC_ThetaDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'LxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaPlus'})
    SxC_ThetaDeltaMinus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaMinus'})
    SxC_ThetaDeltaPlus: SpikeRateTrends = serialized_field(init=False, repr=False, default=None, is_computable=True, hdf_metadata={'track_eXclusive_cells': 'SxC', 'epochs': 'Laps', 'track_change_relative_period': 'DeltaPlus'})
``` 

elements
                                                              

### PyTables approach (`tb.IsDescription`, etc)
The PyTables approach is about defining the `types of each row` in the table for a specific type.
	- Even if you have a list of objects with the same fields, that doesn't mean they can be displayed as a table when they have members of different sizes. They would need scalar members basically.

```python
class NeuronIdentityTable(tb.IsDescription):
    """ represents a single neuron in the scope of multiple sessions for use in a PyTables table or HDF5 output file """
    global_uid = StringCol(16)   # 16-character String, globally unique neuron identifier (across all sessions) composed of a session_uid and the neuron's (session-specific) aclu
    session_uid = StringCol(16)
    ## Session-Local Identifiers
    neuron_id = UInt16Col() # 65535 max neurons
    neuron_type = EnumCol(neuronTypesEnum, 'bad', base='uint8') # 
    shank_index  = UInt16Col() # specific to session
    cluster_index  = UInt16Col() # specific to session
    qclu = UInt8Col() # specific to session
```