---
title: Batch generate context figures over time in pyPhoPlaceCellAnalysis
author: halechr
share_url: https://phohale.sourcegraph.app/deepsearch/15b423d8-8c36-46cb-b495-ec801f37d01e
url: https://phohale.sourcegraph.app/deepsearch/379
created: 2026-06-22T18:24:09Z
updated: 2026-06-22T18:24:55Z
---

# github.com/CommanderPho/pyPhoPlaceCellAnalysis is there a batch way that I generate context figures over time?

Yes, there's a well-established batch pipeline for generating context figures across sessions. Here's a summary of the main paths:

---

## Batch Figure Generation

### 1. Generate Per-Session Figure Scripts (Primary Entry Point)

[`generate_batch_single_session_scripts`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py?L320-L326) in `pythonScriptTemplating.py` is the main way to generate batch figure scripts over time. It creates two scripts per session:

- `run_{session}.py` — loads/computes, **no** figure generation
- `figures_{session}.py` — loads from saved state, **generates figures to file** (`should_perform_figure_generation_to_file=True`)

```python
from pyphoplacecellanalysis.General.Batch.pythonScriptTemplating import generate_batch_single_session_scripts

session_basedirs_dict: Dict[IdentifyingContext, Path] = {
    a_session_folder.context: a_session_folder.path
    for a_session_folder in good_session_concrete_folders
}

output_scripts = generate_batch_single_session_scripts(
    global_data_root_parent_path,
    session_batch_basedirs=session_basedirs_dict,
    included_session_contexts=included_session_contexts,
    output_directory=Path('output/gen_scripts/').resolve(),
    use_separate_run_directories=True,
    create_slurm_scripts=True,           # for GreatLakes
    create_non_slurm_bash_scripts=True,  # for local
    should_generate_figure_scripts=True,
    should_perform_figure_generation_to_file=False,  # figure scripts handle this separately
)
```

The figures scripts are rendered from the [Jinja2 template `python_template.py.j2`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/Resources/Templates/python_template.py.j2?L141-L153) with `should_perform_figure_generation_to_file=True`.

---

### 2. The `BatchSessionCompletionHandler` Trigger

At runtime, each generated script instantiates a [`BatchSessionCompletionHandler`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py?L216-L219) with:

```python
should_perform_figure_generation_to_file=True
should_generate_all_plots=True
```

This calls [`try_complete_figure_generation_to_file`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py?L396-L419), which runs [`main_complete_figure_generations`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/PhoDiba2023Paper.py?L1782-L1784) from `PhoDiba2023Paper.py` — the single top-level function that dispatches all figures.

---

### 3. Interactive Batch Figures (Without Scripts)

For a single loaded pipeline, you can call directly:

```python
from pyphoplacecellanalysis.General.Batch.NonInteractiveProcessing import BatchPlotting

# Set output dir:
collected_figures_folder, fig_out_man = BatchPlotting.find_batch_programmatic_figures_output_dir()

# Generate:
active_identifying_session_ctx, active_out_figures_dict = BatchPlotting.batch_programmatic_figures(curr_active_pipeline)
# or the extended version:
BatchPlotting.batch_extended_programmatic_figures(curr_active_pipeline)
```

See [`BatchPlotting.batch_programmatic_figures`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py?L650-L698) and [`batch_extended_programmatic_figures`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py?L703-L720).

---

### 4. Collect Figures After a Batch Run

After batch figures scripts run (e.g. on GreatLakes), use [`copy_batch_output_figures_to_common_figures_dir`](https://phohale.sourcegraph.app/r/github.com/CommanderPho/pyPhoPlaceCellAnalysis/-/blob/src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py?L1630-L1655) to consolidate all per-session output figures into a single folder:

```python
from pyphoplacecellanalysis.SpecificResults.AcrossSessionResults import copy_batch_output_figures_to_common_figures_dir

_copied_outputs = copy_batch_output_figures_to_common_figures_dir(
    generate_figures_script_paths=generate_figures_script_paths
)
```

Figures land in `gen_scripts/run_{session}/EXTERNAL/Screenshots/ProgrammaticDisplayFunctionTesting/{date}/{session_ctx_path}`.

---

**The typical workflow is:**
1. `generate_batch_single_session_scripts(...)` — generates `run_*.py` + `figures_*.py` + optional SLURM/bash scripts
2. Run the `run_*.py` scripts (compute/save phase)
3. Run the `figures_*.py` scripts (figure generation phase)
4. `copy_batch_output_figures_to_common_figures_dir(...)` — collect all outputs

## Sources

- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ batch context figures generate](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20batch%20context%20figures%20generate)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ context_figure batch](/search?q=patterntype:keyword%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20context_figure%20batch)
- [repo:^github.com/CommanderPho/pyPhoPlaceCellAnalysis$ context figure time over sessions epoch](/search?q=patterntype:nls%20repo:%5Egithub.com%2FCommanderPho%2FpyPhoPlaceCellAnalysis$%20context%20figure%20time%20over%20sessions%20epoch)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py@26f673d?L580-720](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Batch/NonInteractiveProcessing.py?L580-720)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py@26f673d?L390-440](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Batch/BatchJobCompletion/BatchCompletionHandler.py?L390-440)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py@26f673d?L319-500](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/General/Batch/pythonScriptTemplating.py?L319-500)
- [github.com/CommanderPho/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py@26f673d?L1629-1680](/github.com/CommanderPho/pyPhoPlaceCellAnalysis@26f673d14b963201872b14a4d5e35746620c647e/-/blob/src/pyphoplacecellanalysis/SpecificResults/AcrossSessionResults.py?L1629-1680)

