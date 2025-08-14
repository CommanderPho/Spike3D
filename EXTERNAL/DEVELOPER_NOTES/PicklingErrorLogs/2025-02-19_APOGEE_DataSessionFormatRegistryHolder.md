{
	"name": "PicklingError",
	"message": "Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x00000205C2A2CEE0>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell",
	"stack": "---------------------------------------------------------------------------
PicklingError                             Traceback (most recent call last)
Cell In[42], line 1
----> 1 curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE) ## #TODO 2024-02-16 14:25: - [ ] PicklingError: Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x7fd35e66b700>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell
      2 # curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE)
      3 # TypeError: cannot pickle 'traceback' object
      4 # Exception: Can't pickle <enum 'PipelineSavingScheme'>: it's not the same object as pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline.PipelineSavingScheme

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\NeuropyPipeline.py:864, in NeuropyPipeline.save_pipeline(self, saving_mode, active_pickle_filename, override_pickle_path)
    862     saveData(finalized_loaded_sess_pickle_path, db=self) # Save the pipeline out to pickle.
    863 except BaseException as e:
--> 864     raise e
    866 # If we saved to a temporary name, now see if we should overwrite or backup and then replace:
    867 if (is_temporary_file_used and (saving_mode.name == PipelineSavingScheme.TEMP_THEN_OVERWRITE.name)):

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\NeuropyPipeline.py:862, in NeuropyPipeline.save_pipeline(self, saving_mode, active_pickle_filename, override_pickle_path)
    859 # SAVING IS ACTUALLY DONE HERE _______________________________________________________________________________________ #
    860 # Save reloaded pipeline out to pickle for future loading
    861 try:
--> 862     saveData(finalized_loaded_sess_pickle_path, db=self) # Save the pipeline out to pickle.
    863 except BaseException as e:
    864     raise e

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Loading.py:124, in saveData(pkl_path, db, should_append, safe_save)
    113 \"\"\" 
    114 
    115 safe_save: If True, a temporary extension is added to the save path if the file already exists and the file is only overwritten if pickling doesn't throw an exception.
   (...)
    121     saveData('temp.pkl', db)
    122 \"\"\"
    123 if safe_save:
--> 124     safeSaveData(pkl_path, db=db, should_append=should_append)
    125 else:
    126     if should_append:

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Loading.py:76, in safeSaveData(pkl_path, db, should_append, backup_file_if_smaller_than_original, backup_minimum_difference_MB, should_print_output_filesize)
     73 try:
     74     with open(pkl_path, file_mode) as dbfile: 
     75         # source, destination
---> 76         pickle.dump(db, dbfile)
     77         dbfile.close()
     78     # Pickling succeeded
     79 
     80     # If we saved to a temporary name, now see if we should overwrite or backup and then replace:

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:336, in dump(obj, file, protocol, byref, fmode, recurse, **kwds)
    334 _kwds = kwds.copy()
    335 _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
--> 336 Pickler(file, protocol, **_kwds).dump(obj)
    337 return

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:620, in Pickler.dump(self, obj)
    618     raise PicklingError(msg)
    619 else:
--> 620     StockPickler.dump(self, obj)
    621 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:487, in _Pickler.dump(self, obj)
    485 if self.proto >= 4:
    486     self.framer.start_framing()
--> 487 self.save(obj)
    488 self.write(STOP)
    489 self.framer.end_framing()

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError(\"Tuple returned by %s must have \"
    600                         \"two to six elements\" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:717, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715 if state is not None:
    716     if state_setter is None:
--> 717         save(state)
    718         write(BUILD)
    719     else:
    720         # If a state_setter is specified, call it instead of load_build
    721         # to update obj's with its previous state.
    722         # First, push state_setter and its tuple of expected arguments
    723         # (obj, state) onto the stack.

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info(\"# D2\")
   1253 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError(\"Tuple returned by %s must have \"
    600                         \"two to six elements\" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:717, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715 if state is not None:
    716     if state_setter is None:
--> 717         save(state)
    718         write(BUILD)
    719     else:
    720         # If a state_setter is specified, call it instead of load_build
    721         # to update obj's with its previous state.
    722         # First, push state_setter and its tuple of expected arguments
    723         # (obj, state) onto the stack.

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info(\"# D2\")
   1253 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info(\"# D2\")
   1253 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:1002, in _Pickler._batch_setitems(self, items)
   1000     k, v = tmp[0]
   1001     save(k)
-> 1002     save(v)
   1003     write(SETITEM)
   1004 # else tmp is empty, and we're done

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError(\"Tuple returned by %s must have \"
    600                         \"two to six elements\" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:717, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    715 if state is not None:
    716     if state_setter is None:
--> 717         save(state)
    718         write(BUILD)
    719     else:
    720         # If a state_setter is specified, call it instead of load_build
    721         # to update obj's with its previous state.
    722         # First, push state_setter and its tuple of expected arguments
    723         # (obj, state) onto the stack.

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info(\"# D2\")
   1253 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:603, in _Pickler.save(self, obj, save_persistent_id)
    599     raise PicklingError(\"Tuple returned by %s must have \"
    600                         \"two to six elements\" % reduce)
    602 # Save the reduce() output and finally memoize the object
--> 603 self.save_reduce(obj=obj, *rv)

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:687, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    684     raise PicklingError(
    685         \"args[0] from __newobj__ args has the wrong class\")
    686 args = args[1:]
--> 687 save(cls)
    688 save(args)
    689 write(NEWOBJ)

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1838, in save_type(pickler, obj, postproc_list)
   1836             postproc_list = []
   1837         postproc_list.append((setattr, (obj, '__qualname__', obj_name)))
-> 1838     _save_with_postproc(pickler, (_create_type, (
   1839         type(obj), obj.__name__, obj.__bases__, _dict
   1840     )), obj=obj, postproc_list=postproc_list)
   1841     log.info(\"# %s\" % _t)
   1842 else:

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1140, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
   1137     pickler._postproc[id(obj)] = postproc_list
   1139 # TODO: Use state_setter in Python 3.8 to allow for faster cPickle implementations
-> 1140 pickler.save_reduce(*reduction, obj=obj)
   1142 if is_pickler_dill:
   1143     # pickler.x -= 1
   1144     # print(pickler.x*' ', 'pop', obj, id(obj))
   1145     postproc = pickler._postproc.pop(id(obj))

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:692, in _Pickler.save_reduce(self, func, args, state, listitems, dictitems, state_setter, obj)
    690 else:
    691     save(func)
--> 692     save(args)
    693     write(REDUCE)
    695 if obj is not None:
    696     # If the object is already in the memo, this means it is
    697     # recursive. In this case, throw away everything we put on the
    698     # stack, and fetch the object back from the memo.

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:901, in _Pickler.save_tuple(self, obj)
    899 write(MARK)
    900 for element in obj:
--> 901     save(element)
    903 if id(obj) in memo:
    904     # Subtle.  d was not in memo when we entered save_tuple(), so
    905     # the process of saving the tuple's elements must have saved
   (...)
    909     # could have been done in the \"for element\" loop instead, but
    910     # recursive tuples are a rare thing.
    911     get = self.get(memo[id(obj)][0])

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
   1248     if is_dill(pickler, child=False) and pickler._session:
   1249         # we only care about session the first pass thru
   1250         pickler._first_pass = False
-> 1251     StockPickler.save_dict(pickler, obj)
   1252     log.info(\"# D2\")
   1253 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:971, in _Pickler.save_dict(self, obj)
    968     self.write(MARK + DICT)
    970 self.memoize(obj)
--> 971 self._batch_setitems(obj.items())

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1963, in save_function(pickler, obj)
   1960     if state_dict:
   1961         state = state, state_dict
-> 1963     _save_with_postproc(pickler, (_create_function, (
   1964           obj.__code__, globs, obj.__name__, obj.__defaults__,
   1965           closure
   1966     ), state), obj=obj, postproc_list=postproc_list)
   1967 else:
   1968     closure = obj.func_closure

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:1154, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
   1152 if source:
   1153     pickler.write(pickler.get(pickler.memo[id(dest)][0]))
-> 1154     pickler._batch_setitems(iter(source.items()))
   1155 else:
   1156     # Updating with an empty dictionary. Same as doing nothing.
   1157     continue

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:997, in _Pickler._batch_setitems(self, items)
    995     for k, v in tmp:
    996         save(k)
--> 997         save(v)
    998     write(SETITEMS)
    999 elif n:

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\.venv\\lib\\site-packages\\dill\\_dill.py:2004, in save_function(pickler, obj)
   2002     log.info(\"F2: %s\" % obj)
   2003     name = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
-> 2004     StockPickler.save_global(pickler, obj, name=name)
   2005     log.info(\"# F2\")
   2006 return

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:1070, in _Pickler.save_global(self, obj, name)
   1068     obj2, parent = _getattribute(module, name)
   1069 except (ImportError, KeyError, AttributeError):
-> 1070     raise PicklingError(
   1071         \"Can't pickle %r: it's not found as %s.%s\" %
   1072         (obj, module_name, name)) from None
   1073 else:
   1074     if obj2 is not obj:

PicklingError: Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x00000205C2A2CEE0>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell"
}