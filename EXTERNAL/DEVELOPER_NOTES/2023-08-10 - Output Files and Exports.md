
# Pipeline Persistance Properties
```

```


`override_output_basepath`

Probably reuse existing properties instead of adding new ones

`outputs_specifier` # default should perform identical setup to the one now

### Pipeline `_outputs_specifier`
`self._outputs_specifier`


 
# Session:
[/home/halechr/repos/NeuroPy/neuropy/core/session/dataSession.py:287](vscode://file/home/halechr/repos/NeuroPy/neuropy/core/session/dataSession.py:287)
```python
    def get_output_path(self, mkdir_if_needed:bool=True) -> Path:
        """ Build a folder to store the temporary outputs of this session """
        output_data_folder = self.basepath.joinpath('output').resolve()
        if mkdir_if_needed:
            output_data_folder.mkdir(exist_ok=True)
        return output_data_folder
```


# Main call for pipeline to get the output path

Used for global output directory.
[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py:946](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/Stages/Computation.py:946)
```python
 def get_output_path(self) -> Path:
        """ returns the appropriate output path to store the outputs for this session. Usually '$session_folder/outputs/' """
        return self.sess.get_output_path()

```

### Init
[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/NeuropyPipeline.py:286](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/NeuropyPipeline.py:286)
```python
 pipeline_needs_resave = _ensure_unpickled_pipeline_up_to_date(curr_active_pipeline, active_data_mode_name=type_name, basedir=Path(basepath), desired_time_variable_name=desired_time_variable_name, debug_print=debug_print)
            
            curr_active_pipeline._persistance_state = LoadedObjectPersistanceState(finalized_loaded_sess_pickle_path, compare_state_on_load=curr_active_pipeline.pipeline_compare_dict)
            ## Save out the changes to the pipeline after computation to the pickle file for easy loading in the future
            if pipeline_needs_resave:
                if not skip_save_on_initial_load:
                    curr_active_pipeline.save_pipeline(active_pickle_filename=active_pickle_filename)
                else:
                    if progress_print:
                        print(f'pipeline_needs_resave but skip_save_on_initial_load == True, so saving will be skipped entirely. Be sure to save manually if there are changes.')
            else:
                if progress_print:
                    print(f'property already present in pickled version. No need to save.')
```


### Properties
[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/NeuropyPipeline.py:426](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Pipeline/NeuropyPipeline.py:426)
```python
# Persistance and Saving _____________________________________________________________________________________________ #
    @property
    def pipeline_compare_dict(self):
        """The pipeline_compare_dict property."""
        return NeuropyPipeline.build_pipeline_compare_dict(self)

    @property
    def persistance_state(self):
        """The persistance_state property."""
        return self._persistance_state
    
    @property
    def pickle_path(self):
        """ indicates that this pipeline doesn't have a corresponding pickle file that it was loaded from"""
        if self.persistance_state is None:
            return None
        else:
            return self.persistance_state.file_path
```


# Existing `FigureOutputManager` approach
[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:158](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:158)
```python
class FigureOutputLocation(Enum):
    """Specifies the filesystem location for the parent folder where figures are output."""
    DAILY_PROGRAMMATIC_OUTPUT_FOLDER = "daily_programmatic_output_folder" # the common folder for today's date
    SESSION_OUTPUT_FOLDER = "session_output_folder" # the session-specific output folder. f"{session_path}/output/figures"
    CUSTOM = "custom" # other folder. Must be specified.
    
```

[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:202](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:202)
```python

class ContextToPathMode(Enum):
    """ Controls how hierarchical contexts (IdentityContext) are mapped to relative output paths.
    In HIERARCHY_UNIQUE mode the folder hierarchy partially specifies the context (mainly the session part, e.g. './kdiba/gor01/two/2006-6-08_21-16-25/') so the filenames don't need to be completely unique (they can drop the 'kdiba_gor01_two_2006-6-08_21-16-25_' portion)
        'output/kdiba/gor01/two/2006-6-08_21-16-25/batch_pho_jonathan_replay_firing_rate_comparison.png

    In GLOBAL_UNIQUE mode the outputs are placed in a flat folder structure ('output/'), meaning the filenames need to be completely unique and specify all parts of the context:
        'output/kdiba_gor01_two_2006-6-08_21-16-25_batch_pho_jonathan_replay_firing_rate_comparison.png'
    """
    HIERARCHY_UNIQUE = "hierarchy_unique"
    GLOBAL_UNIQUE = "global_unique"
```


[/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:278](vscode://file/home/halechr/repos/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Mixins/ExportHelpers.py:278)
```python
class FigureOutputManager:
    """ 2023-06-14 - Manages figure output. Singleton/not persisted.

    Usage:
        fig_man = FigureOutputManager(figure_output_location=FigureOutputLocation.DAILY_PROGRAMMATIC_OUTPUT_FOLDER, context_to_path_mode=ContextToPathMode.GLOBAL_UNIQUE)
        test_context = IdentifyingContext(format_name='kdiba',animal='gor01',exper_name='one',session_name='2006-6-08_14-26-15',display_fn_name='display_long_short_laps')
        fig_man.get_figure_save_file_path(test_context, make_folder_if_needed=False)
        >>> Path('/home/halechr/repo/Spike3D/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/2023-06-14/kdiba_gor01_one_2006-6-08_14-26-15_display_long_short_laps')
    """
    figure_output_location: FigureOutputLocation
    context_to_path_mode: ContextToPathMode
```

