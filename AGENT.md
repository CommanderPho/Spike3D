# AGENT.md - Development Guidelines for Spike3D

## Build/Test Commands
- **Build**: `uv sync` or `uv install` - Install dependencies using uv package manager
- **Type Check**: `pyright` - Type checking (configured in pyrightconfig.json, Python 3.9)
- **Code Format**: `black` - Code formatting (available in dev dependencies)
- **Tests**: `pytest` - Run tests (primarily in EXTERNAL/TESTING/ and scattered test files)
- **Single Test**: `pytest path/to/test_file.py::test_function_name`
- **Lint**: `mypy` - Static type analysis (available in dev dependencies)

## Architecture & Structure
- **Main Project**: Spike3D - 3D neural data visualization and analysis
- **Core Subprojects**: neuropy, pyphocorehelpers, pyphoplacecellanalysis (editable local dependencies)
- **GUI Framework**: PyQt5 with custom widgets, 3D visualization via PyVista/VTK
- **Data Storage**: HDF5 (h5py, tables), with custom HDF_SerializationMixin
- **Key Directories**: GUI/Qt/ (UI components), Analysis/ (data processing), Pho3D/ (3D visualization)

## Code Style & Conventions
- **Classes**: PascalCase (e.g., `Neurons`, `BinnedSpiketrain`)
- **Functions/Variables**: snake_case (e.g., `get_by_id`, `neuron_ids`)
- **Type Hints**: Required with comprehensive annotations using typing, nptyping, TypeAlias
- **Imports**: Standard library → Third-party → Local (relative imports last)
- **Error Handling**: Assert for validation, try-except with detailed context, custom CapturedException class
- **Documentation**: Comprehensive docstrings with Parameters/Returns sections
- **Decorators**: Use @metadata_attributes for analysis functions with tags and dependencies
