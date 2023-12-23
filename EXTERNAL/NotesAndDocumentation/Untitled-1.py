{
	"name": "TypeError",
	"message": "cannot pickle 'QApplication' object",
	"stack": "---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
c:\\Users\\pho\\repos\\Spike3DWorkEnv\\Spike3D\\ReviewOfWork_2023-12-22.ipynb Cell 98 line 1
      <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=4'>5</a> dill.detect.trace(True)
      <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=5'>6</a> # with dill.detect.trace():
      <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=6'>7</a> # saveData(rank_order_output_path, (curr_active_pipeline.global_computation_results.computed_data['RankOrder'].__dict__,))
      <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=7'>8</a> # custom_dumps(rank_order_results.ripple_new_output_tuple)
   (...)
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=11'>12</a> # \tcustom_dump(db, dbfile) # ModuleExcludesPickler
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=12'>13</a> # \tdbfile.close()
---> <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=14'>15</a> dill.dumps(rank_order_results.ripple_new_output_tuple)
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=15'>16</a> # custom_dump(db, dbfile) # ModuleExcludesPickler
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=16'>17</a> # saveData(rank_order_output_path, (curr_active_pipeline.global_computation_results.computed_data['RankOrder'].__dict__,))
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=17'>18</a> 
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=18'>19</a> 
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=19'>20</a> 
     <a href='vscode-notebook-cell:/c%3A/Users/pho/repos/Spike3DWorkEnv/Spike3D/ReviewOfWork_2023-12-22.ipynb#Z1120sZmlsZQ%3D%3D?line=20'>21</a> # saveData('output/test_ripple_new_output_tuple.pkl', (rank_order_results.ripple_new_output_tuple,))

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:364, in dumps(obj, protocol, byref, fmode, recurse, **kwds)
    340 \"\"\"
    341 Pickle an object to a string.
    342 
   (...)
    361 Default values for keyword arguments can be set in :mod:`dill.settings`.
    362 \"\"\"
    363 file = StringIO()
--> 364 dump(obj, file, protocol, byref, fmode, recurse, **kwds)#, strictio)
    365 return file.getvalue()

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:336, in dump(obj, file, protocol, byref, fmode, recurse, **kwds)
    334 _kwds = kwds.copy()
    335 _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
--> 336 Pickler(file, protocol, **_kwds).dump(obj)
    337 return

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:620, in Pickler.dump(self, obj)
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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:886, in _Pickler.save_tuple(self, obj)
    884 if n <= 3 and self.proto >= 2:
    885     for element in obj:
--> 886         save(element)
    887     # Subtle.  Same as in the big comment below.
    888     if id(obj) in memo:

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:560, in _Pickler.save(self, obj, save_persistent_id)
    558 f = self.dispatch.get(t)
    559 if f is not None:
--> 560     f(self, obj)  # Call unbound method with explicit self
    561     return
    563 # Check private dispatch table if any, or else
    564 # copyreg.dispatch_table

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:931, in _Pickler.save_list(self, obj)
    928     self.write(MARK + LIST)
    930 self.memoize(obj)
--> 931 self._batch_appends(obj)

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:955, in _Pickler._batch_appends(self, items)
    953     write(MARK)
    954     for x in tmp:
--> 955         save(x)
    956     write(APPENDS)
    957 elif n:

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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1838, in save_type(pickler, obj, postproc_list)
   1836             postproc_list = []
   1837         postproc_list.append((setattr, (obj, '__qualname__', obj_name)))
-> 1838     _save_with_postproc(pickler, (_create_type, (
   1839         type(obj), obj.__name__, obj.__bases__, _dict
   1840     )), obj=obj, postproc_list=postproc_list)
   1841     log.info(\"# %s\" % _t)
   1842 else:

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1140, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1963, in save_function(pickler, obj)
   1960     if state_dict:
   1961         state = state, state_dict
-> 1963     _save_with_postproc(pickler, (_create_function, (
   1964           obj.__code__, globs, obj.__name__, obj.__defaults__,
   1965           closure
   1966     ), state), obj=obj, postproc_list=postproc_list)
   1967 else:
   1968     closure = obj.func_closure

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1154, in _save_with_postproc(pickler, reduction, is_pickler_dill, obj, postproc_list)
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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1765, in save_module(pickler, obj)
   1762     _main_dict = obj.__dict__.copy() #XXX: better no copy? option to copy?
   1763     [_main_dict.pop(item, None) for item in singletontypes
   1764         + [\"__builtins__\", \"__loader__\"]]
-> 1765     pickler.save_reduce(_import_module, (obj.__name__,), obj=obj,
   1766                         state=_main_dict)
   1767     log.info(\"# M1\")
   1768 elif PY3 and obj.__name__ == \"dill._dill\":

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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1765, in save_module(pickler, obj)
   1762     _main_dict = obj.__dict__.copy() #XXX: better no copy? option to copy?
   1763     [_main_dict.pop(item, None) for item in singletontypes
   1764         + [\"__builtins__\", \"__loader__\"]]
-> 1765     pickler.save_reduce(_import_module, (obj.__name__,), obj=obj,
   1766                         state=_main_dict)
   1767     log.info(\"# M1\")
   1768 elif PY3 and obj.__name__ == \"dill._dill\":

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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1765, in save_module(pickler, obj)
   1762     _main_dict = obj.__dict__.copy() #XXX: better no copy? option to copy?
   1763     [_main_dict.pop(item, None) for item in singletontypes
   1764         + [\"__builtins__\", \"__loader__\"]]
-> 1765     pickler.save_reduce(_import_module, (obj.__name__,), obj=obj,
   1766                         state=_main_dict)
   1767     log.info(\"# M1\")
   1768 elif PY3 and obj.__name__ == \"dill._dill\":

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

File k:\\FastSwap\\Environments\\pypoetry\\pypoetry\\Cache\\virtualenvs\\spike3d-UP7QTzFM-py3.9\\lib\\site-packages\\dill\\_dill.py:1251, in save_module_dict(pickler, obj)
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

File ~\\.pyenv\\pyenv-win\\versions\\3.9.13\\lib\\pickle.py:578, in _Pickler.save(self, obj, save_persistent_id)
    576 reduce = getattr(obj, \"__reduce_ex__\", None)
    577 if reduce is not None:
--> 578     rv = reduce(self.proto)
    579 else:
    580     reduce = getattr(obj, \"__reduce__\", None)

TypeError: cannot pickle 'QApplication' object"
}