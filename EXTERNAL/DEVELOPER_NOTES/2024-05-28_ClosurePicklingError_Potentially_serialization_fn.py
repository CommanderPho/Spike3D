
NOTE: This occured only after adding:
```
@define(slots=False, repr=False, eq=False)
class WCorrShuffle(ComputedResult):
```

And changing fields to `non_serialized_field`, `serialized_field`, and `serialized_attribute_field`



# ==================================================================================================================== #
# Error                                                                                                                #
# ==================================================================================================================== #

{
	"name": "PicklingError",
	"message": "Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x0000014B4DB900D0>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell",
	"stack": "---------------------------------------------------------------------------
PicklingError                             Traceback (most recent call last)
Cell In[74], line 3
      1 from pyphoplacecellanalysis.General.Pipeline.Stages.Loading import saveData
----> 3 saveData('test_wcorr_tool_modern.pkl', (wcorr_tool, ))

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Loading.py:116, in saveData(pkl_path, db, should_append, safe_save)
    105 \"\"\" 
    106 
    107 safe_save: If True, a temporary extension is added to the save path if the file already exists and the file is only overwritten if pickling doesn't throw an exception.
   (...)
    113     saveData('temp.pkl', db)
    114 \"\"\"
    115 if safe_save:
--> 116     safeSaveData(pkl_path, db=db, should_append=should_append)
    117 else:
    118     if should_append:

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Loading.py:98, in safeSaveData(pkl_path, db, should_append, backup_file_if_smaller_than_original, backup_minimum_difference_MB)
     96 if is_temporary_file_used:
     97     pkl_path.unlink(missing_ok=True) # removes the incomplete file. The user's file located at _desired_final_pickle_path is still intact.
---> 98 raise e

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\Stages\\Loading.py:74, in safeSaveData(pkl_path, db, should_append, backup_file_if_smaller_than_original, backup_minimum_difference_MB)
     71 try:
     72     with open(pkl_path, file_mode) as dbfile: 
     73         # source, destination
---> 74         pickle.dump(db, dbfile)
     75         dbfile.close()
     76     # Pickling succeeded
     77 
     78     # If we saved to a temporary name, now see if we should overwrite or backup and then replace:

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:336, in dump(obj, file, protocol, byref, fmode, recurse, **kwds)
    334 _kwds = kwds.copy()
    335 _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
--> 336 Pickler(file, protocol, **_kwds).dump(obj)
    337 return

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:620, in Pickler.dump(self, obj)
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

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:886, in _Pickler.save_tuple(self, obj)
    884 if n <= 3 and self.proto >= 2:
    885     for element in obj:
--> 886         save(element)
    887     # Subtle.  Same as in the big comment below.
    888     if id(obj) in memo:

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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1838, in save_type(pickler, obj, postproc_list)
   1836             postproc_list = []
   1837         postproc_list.append((setattr, (obj, '__qualname__', obj_name)))
-> 1838     _save_with_postproc(pickler, (_create_type, (
   1839         type(obj), obj.__name__, obj.__bases__, _dict
   1840     )), obj=obj, postproc_list=postproc_list)
   1841     log.info(\"# %s\" % _t)
   1842 else:

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1140, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1838, in save_type(pickler, obj, postproc_list)
   1836             postproc_list = []
   1837         postproc_list.append((setattr, (obj, '__qualname__', obj_name)))
-> 1838     _save_with_postproc(pickler, (_create_type, (
   1839         type(obj), obj.__name__, obj.__bases__, _dict
   1840     )), obj=obj, postproc_list=postproc_list)
   1841     log.info(\"# %s\" % _t)
   1842 else:

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1140, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1963, in save_function(pickler, obj)
   1960     if state_dict:
   1961         state = state, state_dict
-> 1963     _save_with_postproc(pickler, (_create_function, (
   1964           obj.__code__, globs, obj.__name__, obj.__defaults__,
   1965           closure
   1966     ), state), obj=obj, postproc_list=postproc_list)
   1967 else:
   1968     closure = obj.func_closure

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:1154, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
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

File k:\\FastSwap\\AppData\\VSCode\\black\\.venv_black\\lib\\site-packages\\dill\\_dill.py:2004, in save_function(pickler, obj)
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

PicklingError: Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x0000014B4DB900D0>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell"
}