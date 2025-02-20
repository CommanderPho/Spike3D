{
	"name": "TypeError",
	"message": "unhashable type: 'list'",
	"stack": "---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[59], line 1
----> 1 curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.TEMP_THEN_OVERWRITE) ## #TODO 2024-02-16 14:25: - [ ] PicklingError: Can't pickle <function make_set_closure_cell.<locals>.set_closure_cell at 0x7fd35e66b700>: it's not found as attr._compat.make_set_closure_cell.<locals>.set_closure_cell
      2 # curr_active_pipeline.save_pipeline(saving_mode=PipelineSavingScheme.OVERWRITE_IN_PLACE)
      3 # TypeError: cannot pickle 'traceback' object
      4 # Exception: Can't pickle <enum 'PipelineSavingScheme'>: it's not the same object as pyphoplacecellanalysis.General.Pipeline.NeuropyPipeline.PipelineSavingScheme

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\NeuropyPipeline.py:887, in NeuropyPipeline.save_pipeline(self, saving_mode, active_pickle_filename, override_pickle_path)
    883     finalized_loaded_sess_pickle_path = _desired_finalized_loaded_sess_pickle_path
    885 if not used_existing_pickle_path:
    886     # the pickle path changed, so set it on the pipeline:
--> 887     self._persistance_state = LoadedObjectPersistanceState(finalized_loaded_sess_pickle_path, compare_state_on_load=self.pipeline_compare_dict)
    889 self.logger.info(f'\\t save complete.')
    890 return finalized_loaded_sess_pickle_path

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\NeuropyPipeline.py:624, in NeuropyPipeline.pipeline_compare_dict(self)
    621 @property
    622 def pipeline_compare_dict(self):
    623     \"\"\"The pipeline_compare_dict property.\"\"\"
--> 624     return NeuropyPipeline.build_pipeline_compare_dict(self)

File ~\\repos\\Spike3DWorkEnv\\pyPhoPlaceCellAnalysis\\src\\pyphoplacecellanalysis\\General\\Pipeline\\NeuropyPipeline.py:578, in NeuropyPipeline.build_pipeline_compare_dict(cls, a_pipeline)
    564         comp_config_results_list = None
    566     # ## Add the global_computation_results to the comp_config_results_list with a common key '__GLOBAL__':
    567     # if hasattr(a_pipeline, 'global_computation_results'):
    568     #     if comp_config_results_list is None:
   (...)
    573     #     # has none
    574     #     pass # 
    576     out_results_dict.update(active_completed_computation_result_names = tuple(a_pipeline.active_completed_computation_result_names), # ['maze1_PYR', 'maze2_PYR', 'maze_PYR']
    577         active_incomplete_computation_result_status_dicts= freeze(a_pipeline.active_incomplete_computation_result_status_dicts),
--> 578         computation_result_computed_data_names = freeze(comp_config_results_list)
    579     )
    581 return out_results_dict

File ~\\repos\\Spike3DWorkEnv\\pyPhoCoreHelpers\\src\\pyphocorehelpers\\hashing_helpers.py:5, in freeze(d)
      3 \"\"\" recursively freezes dicts with nested dict/list elements so they may be hashed \"\"\"
      4 if isinstance(d, dict):
----> 5     return frozenset((key, freeze(value)) for key, value in d.items())
      6 elif isinstance(d, list):
      7     return tuple(freeze(value) for value in d)

File ~\\repos\\Spike3DWorkEnv\\pyPhoCoreHelpers\\src\\pyphocorehelpers\\hashing_helpers.py:5, in <genexpr>(.0)
      3 \"\"\" recursively freezes dicts with nested dict/list elements so they may be hashed \"\"\"
      4 if isinstance(d, dict):
----> 5     return frozenset((key, freeze(value)) for key, value in d.items())
      6 elif isinstance(d, list):
      7     return tuple(freeze(value) for value in d)

File ~\\repos\\Spike3DWorkEnv\\pyPhoCoreHelpers\\src\\pyphocorehelpers\\hashing_helpers.py:5, in freeze(d)
      3 \"\"\" recursively freezes dicts with nested dict/list elements so they may be hashed \"\"\"
      4 if isinstance(d, dict):
----> 5     return frozenset((key, freeze(value)) for key, value in d.items())
      6 elif isinstance(d, list):
      7     return tuple(freeze(value) for value in d)

File ~\\repos\\Spike3DWorkEnv\\NeuroPy\
europy\\utils\\dynamic_container.py:135, in DynamicContainer.__hash__(self)
    133 values_tuple = list(self.values())
    134 combined_tuple = tuple(member_names_tuple + values_tuple)
--> 135 return hash(combined_tuple)

File ~\\repos\\Spike3DWorkEnv\\NeuroPy\
europy\\analyses\\placefields.py:232, in PlacefieldComputationParameters.__hash__(self)
    230 values_tuple = list(dict_rep.values())
    231 combined_tuple = tuple(member_names_tuple + values_tuple)
--> 232 return hash(combined_tuple)

TypeError: unhashable type: 'list'"
}