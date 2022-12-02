basedir: /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15
finalized_loaded_sess_pickle_path: /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/loadedSessPickle.pkl
Loading loaded session pickle file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/loadedSessPickle.pkl... done.
Failure loading /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/loadedSessPickle.pkl.
Must reload/rebuild.
NeuropyPipeline.on_stage_changed(new_stage="PipelineStage.Input")
Loading matlab import file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.epochs_info.mat... done.
Loading matlab import file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.position_info.mat... done.
Loading matlab import file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.spikes.mat... 
/Users/pho/repo/Python Projects/NeuroPy/neuropy/core/session/Formats/SessionSpecifications.py:139: UserWarning: WARNING: Optional File: /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.dat does not exist. Continuing without it.
  warnings.warn(f'WARNING: Optional File: {an_optional_filepath} does not exist. Continuing without it.')
done.
Failure loading .position.npy. Must recompute.

Computing linear positions for all active epochs for session... Saving updated position results results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.position.npy... 2006-6-08_14-26-15.position.npy saved
done.
	 force_recompute is True! Forcing recomputation of .interpolated_spike_positions.npy

Computing interpolate_spike_positions columns results to spikes_df... done.
	 Saving updated interpolated spike position results results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.interpolated_spike_positions.npy... 2006-6-08_14-26-15.interpolated_spike_positions.npy saved
done.
Loading matlab import file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.laps_info.mat... done.
setting laps object.
session.laps loaded successfully!
Loading matlab import file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/2006-6-08_14-26-15.replay_info.mat... done.
session.replays loaded successfully!
Loading success: .ripple.npy.
Loading success: .mua.npy.
Loading success: .pbe.npy.
Computing spikes_df PBEs column results to spikes_df... done.
Computing added spike scISI column results to spikes_df... done.
NeuropyPipeline.on_stage_changed(new_stage="PipelineStage.Loaded")
Saving (file mode 'w+b') saved session pickle file results to /Volumes/MoverNew/data/KDIBA/gor01/one/2006-6-08_14-26-15/loadedSessPickle.pkl... done.
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/var/folders/2q/1j2p9lpn4gjd4n6zs3wkgh080000gn/T/ipykernel_22473/1958510707.py in <module>
     29 print(f'basedir: {str(basedir)}')
     30 
---> 31 curr_active_pipeline = NeuropyPipeline.try_init_from_saved_pickle_or_reload_if_needed(active_data_mode_name, active_data_mode_type_properties, override_basepath=Path(basedir), override_post_load_functions=[], force_reload=False, active_pickle_filename='loadedSessPickle.pkl', skip_save=False)
     32 # active_session_filter_configurations = active_data_mode_registered_class.build_default_filter_functions(sess=curr_active_pipeline.sess) # build_filters_pyramidal_epochs(sess=curr_kdiba_pipeline.sess)
     33 active_session_filter_configurations = active_data_mode_registered_class.build_filters_pyramidal_epochs(sess=curr_active_pipeline.sess, epoch_name_whitelist=['maze','maze1','maze2'])

~/repo/Python Projects/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/NeuropyPipeline.py in try_init_from_saved_pickle_or_reload_if_needed(cls, type_name, known_type_properties, override_basepath, override_post_load_functions, force_reload, active_pickle_filename, skip_save)
    229             # Save reloaded pipeline out to pickle for future loading
    230             if not skip_save:
--> 231                 saveData(finalized_loaded_sess_pickle_path, db=curr_active_pipeline) # 589 MB
    232             else:
    233                 print('skip_save is True so resultant pipeline will not be saved to the pickle file.')

~/repo/Python Projects/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Loading.py in saveData(pkl_path, db, should_append)
     28         with open(pkl_path, file_mode) as dbfile:
     29             # source, destination
---> 30             pickle.dump(db, dbfile)
     31             dbfile.close()
     32 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in dump(obj, file, protocol, byref, fmode, recurse, **kwds)
    233     _kwds = kwds.copy()
    234     _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
--> 235     Pickler(file, protocol, **_kwds).dump(obj)
    236     return
    237 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in dump(self, obj)
    392     def dump(self, obj): #NOTE: if settings change, need to update attributes
    393         logger.trace_setup(self)
--> 394         StockPickler.dump(self, obj)
    395 
    396     dump.__doc__ = StockPickler.dump.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in dump(self, obj)
    485         if self.proto >= 4:
    486             self.framer.start_framing()
--> 487         self.save(obj)
    488         self.write(STOP)
    489         self.framer.end_framing()

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    601 
    602         # Save the reduce() output and finally memoize the object
--> 603         self.save_reduce(obj=obj, *rv)
    604 
    605     def persistent_id(self, obj):

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715         if state is not None:
    716             if state_setter is None:
--> 717                 save(state)
    718                 write(BUILD)
    719             else:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_module_dict(pickler, obj)
   1184             # we only care about session the first pass thru
   1185             pickler._first_pass = False
-> 1186         StockPickler.save_dict(pickler, obj)
   1187         logger.trace(pickler, "# D2")
   1188     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_dict(self, obj)
    969 
    970         self.memoize(obj)
--> 971         self._batch_setitems(obj.items())
    972 
    973     dispatch[dict] = save_dict

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in _batch_setitems(self, items)
    995                 for k, v in tmp:
    996                     save(k)
--> 997                     save(v)
    998                 write(SETITEMS)
    999             elif n:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    601 
    602         # Save the reduce() output and finally memoize the object
--> 603         self.save_reduce(obj=obj, *rv)
    604 
    605     def persistent_id(self, obj):

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715         if state is not None:
    716             if state_setter is None:
--> 717                 save(state)
    718                 write(BUILD)
    719             else:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_module_dict(pickler, obj)
   1184             # we only care about session the first pass thru
   1185             pickler._first_pass = False
-> 1186         StockPickler.save_dict(pickler, obj)
   1187         logger.trace(pickler, "# D2")
   1188     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_dict(self, obj)
    969 
    970         self.memoize(obj)
--> 971         self._batch_setitems(obj.items())
    972 
    973     dispatch[dict] = save_dict

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in _batch_setitems(self, items)
    995                 for k, v in tmp:
    996                     save(k)
--> 997                     save(v)
    998                 write(SETITEMS)
    999             elif n:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_module_dict(pickler, obj)
   1184             # we only care about session the first pass thru
   1185             pickler._first_pass = False
-> 1186         StockPickler.save_dict(pickler, obj)
   1187         logger.trace(pickler, "# D2")
   1188     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_dict(self, obj)
    969 
    970         self.memoize(obj)
--> 971         self._batch_setitems(obj.items())
    972 
    973     dispatch[dict] = save_dict

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in _batch_setitems(self, items)
   1000                 k, v = tmp[0]
   1001                 save(k)
-> 1002                 save(v)
   1003                 write(SETITEM)
   1004             # else tmp is empty, and we're done

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    601 
    602         # Save the reduce() output and finally memoize the object
--> 603         self.save_reduce(obj=obj, *rv)
    604 
    605     def persistent_id(self, obj):

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715         if state is not None:
    716             if state_setter is None:
--> 717                 save(state)
    718                 write(BUILD)
    719             else:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_module_dict(pickler, obj)
   1184             # we only care about session the first pass thru
   1185             pickler._first_pass = False
-> 1186         StockPickler.save_dict(pickler, obj)
   1187         logger.trace(pickler, "# D2")
   1188     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_dict(self, obj)
    969 
    970         self.memoize(obj)
--> 971         self._batch_setitems(obj.items())
    972 
    973     dispatch[dict] = save_dict

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in _batch_setitems(self, items)
    995                 for k, v in tmp:
    996                     save(k)
--> 997                     save(v)
    998                 write(SETITEMS)
    999             elif n:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    601 
    602         # Save the reduce() output and finally memoize the object
--> 603         self.save_reduce(obj=obj, *rv)
    604 
    605     def persistent_id(self, obj):

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715         if state is not None:
    716             if state_setter is None:
--> 717                 save(state)
    718                 write(BUILD)
    719             else:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_module_dict(pickler, obj)
   1184             # we only care about session the first pass thru
   1185             pickler._first_pass = False
-> 1186         StockPickler.save_dict(pickler, obj)
   1187         logger.trace(pickler, "# D2")
   1188     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_dict(self, obj)
    969 
    970         self.memoize(obj)
--> 971         self._batch_setitems(obj.items())
    972 
    973     dispatch[dict] = save_dict

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in _batch_setitems(self, items)
    995                 for k, v in tmp:
    996                     save(k)
--> 997                     save(v)
    998                 write(SETITEMS)
    999             elif n:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_numpy_array(pickler, obj)
    379                     npdict = getattr(obj, '__dict__', None)
    380                     f, args, state = obj.__reduce__()
--> 381                     pickler.save_reduce(_create_array, (f,args,state,npdict), obj=obj)
    382                     logger.trace(pickler, "# Nu")
    383                     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    690         else:
    691             save(func)
--> 692             save(args)
    693             write(REDUCE)
    694 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_tuple(self, obj)
    899         write(MARK)
    900         for element in obj:
--> 901             save(element)
    902 
    903         if id(obj) in memo:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    558             f = self.dispatch.get(t)
    559             if f is not None:
--> 560                 f(self, obj)  # Call unbound method with explicit self
    561                 return
    562 

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save_module_dict(pickler, obj)
   1184             # we only care about session the first pass thru
   1185             pickler._first_pass = False
-> 1186         StockPickler.save_dict(pickler, obj)
   1187         logger.trace(pickler, "# D2")
   1188     return

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save_dict(self, obj)
    969 
    970         self.memoize(obj)
--> 971         self._batch_setitems(obj.items())
    972 
    973     dispatch[dict] = save_dict

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in _batch_setitems(self, items)
    995                 for k, v in tmp:
    996                     save(k)
--> 997                     save(v)
    998                 write(SETITEMS)
    999             elif n:

~/mambaforge/envs/viz3d/lib/python3.9/site-packages/dill/_dill.py in save(self, obj, save_persistent_id)
    386             msg = "Can't pickle %s: attribute lookup builtins.generator failed" % GeneratorType
    387             raise PicklingError(msg)
--> 388         StockPickler.save(self, obj, save_persistent_id)
    389 
    390     save.__doc__ = StockPickler.save.__doc__

~/mambaforge/envs/viz3d/lib/python3.9/pickle.py in save(self, obj, save_persistent_id)
    576                 reduce = getattr(obj, "__reduce_ex__", None)
    577                 if reduce is not None:
--> 578                     rv = reduce(self.proto)
    579                 else:
    580                     reduce = getattr(obj, "__reduce__", None)

TypeError: cannot pickle 'mmap.mmap' object



TypeError: cannot pickle 'mmap.mmap' object ' #11823
Issue with macOS (Darwin) and multi-processing
https://github.com/stamparm/maltrail/issues/11823


# PROGRESS: The issue has been traced down to `BinarysignalIO._raw_traces` which is a np.nmap object
Specifically
```python
    self.session.eegfile._raw_traces
    self.session.datfile._raw_traces
```

```python
@classmethod
    def _load_eegfile(cls, filepath, session):
        # .eegfile
        try:
            session.eegfile = BinarysignalIO(filepath, n_channels=session.recinfo.n_channels, sampling_rate=session.recinfo.eeg_sampling_rate)
        except ValueError:
            print('session.recinfo.eeg_filename exists ({}) but file cannot be loaded in the appropriate format. Skipping. \n'.format(filepath))
            session.eegfile = None
        return session

    @classmethod
    def _load_datfile(cls, filepath, session):
        # .datfile
        if filepath.is_file():
            session.datfile = BinarysignalIO(filepath, n_channels=session.recinfo.n_channels, sampling_rate=session.recinfo.dat_sampling_rate)
        else:
            session.datfile = None   
        return session
```

# Solution: Add custom `__getstate__`/`__setstate__` to `BinarysignalIO` that removes the `_raw_traces` on save and reloads them after loading


