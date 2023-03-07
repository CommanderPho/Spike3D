---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[9], line 1
----> 1 curr_active_pipeline.save_pipeline()

File ~\repos\IsolatedSpike3DEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\NeuropyPipeline.py:606, in NeuropyPipeline.save_pipeline(self, saving_mode, active_pickle_filename)
    602         self.logger.warning(f'WARNING: saving_mode is OVERWRITE_IN_PLACE so {finalized_loaded_sess_pickle_path} will be overwritten even though exists.')
    605 # Save reloaded pipeline out to pickle for future loading
--> 606 saveData(finalized_loaded_sess_pickle_path, db=self) # Save the pipeline out to pickle.
    608 # If we saved to a temporary name, now see if we should overwrite or backup and then replace:
    609 if saving_mode.name == PipelineSavingScheme.TEMP_THEN_OVERWRITE.name:

File ~\repos\IsolatedSpike3DEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\General\Pipeline\Stages\Loading.py:30, in saveData(pkl_path, db, should_append)
     27 with ProgressMessagePrinter(pkl_path, f"Saving (file mode '{file_mode}')", 'saved session pickle file'):
     28     with open(pkl_path, file_mode) as dbfile: 
     29         # source, destination
---> 30         pickle.dump(db, dbfile)
     31         dbfile.close()

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:336, in dump(obj, file, protocol, byref, fmode, recurse, **kwds)
    334 _kwds = kwds.copy()
    335 _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
--> 336 Pickler(file, protocol, **_kwds).dump(obj)
    337 return

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:620, in Pickler.dump(self, obj)
    618     raise PicklingError(msg)
    619 else:
--> 620     StockPickler.dump(self, obj)
    621 return

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:487, in _Pickler.dump(self, obj)
    485 if self.proto >= 4:
    486     self.framer.start_framing()
--> 487 self.save(obj)
    488 self.write(STOP)
    489 self.framer.end_framing()

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError("Tuple returned by %s must have "
    600                         "two to six elements" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:717, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715 if state is not None:
    716     if state_setter is None:
--> 717         save(state)
    718         write(BUILD)
    719     else:
    720         # If a state_setter is specified, call it instead of load_build
    721         # to update obj's with its previous state.
    722         # First, push state_setter and its tuple of expected arguments
    723         # (obj, state) onto the stack.

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info("# D2")
   1253 return

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError("Tuple returned by %s must have "
    600                         "two to six elements" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:717, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715 if state is not None:
    716     if state_setter is None:
--> 717         save(state)
    718         write(BUILD)
    719     else:
    720         # If a state_setter is specified, call it instead of load_build
    721         # to update obj's with its previous state.
    722         # First, push state_setter and its tuple of expected arguments
    723         # (obj, state) onto the stack.

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info("# D2")
   1253 return

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError("Tuple returned by %s must have "
    600                         "two to six elements" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:717, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715 if state is not None:
    716     if state_setter is None:
--> 717         save(state)
    718         write(BUILD)
    719     else:
    720         # If a state_setter is specified, call it instead of load_build
    721         # to update obj's with its previous state.
    722         # First, push state_setter and its tuple of expected arguments
    723         # (obj, state) onto the stack.

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info("# D2")
   1253 return

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info("# D2")
   1253 return

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\AppData\Local\pypoetry\Cache\virtualenvs\spike3d-WtdfU5rp-py3.9\lib\site-packages\dill\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info("# D2")
   1253 return

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\.pyenv\pyenv-win\versions\3.9.13\lib\pickle.py:578, in _Pickler.save(self, obj, save_persistent_id)
    576 reduce = getattr(obj, "__reduce_ex__", None)
    577 if reduce is not None:
--> 578     rv = reduce(self.proto)
    579 else:
    580     reduce = getattr(obj, "__reduce__", None)

TypeError: cannot pickle 'Spike2DRaster' object