
conda install -c conda-forge jupyterlab


conda cr
mamba create -n viz3d matplotlib seaborn pymc jupyterlab pyvista vaex ipympl hdf5storage ipywidgets ipygany ipyvolume ipyvtklink panel -c conda-forge


# Updated 2022-03-08 Environment Install
## Creates the new environment from the .yml file:
mamba env create -f environment_from_history_pruned.yml

## Activate the new environment:
conda activate phoviz_ultimate

## Install the packages that couldn't be installed on the first wave:
mamba install pyopengl cupy cudatoolkit numba panel pingouin -c conda-forge
### Note: pingouin seems to only be used in one file. panel is being phased out.

# Clone the required direct dependency repos and install using `pip install -e .`
git clone https://github.com/CommanderPho/NeuroPy.git
git clone https://github.com/CommanderPho/pyPhoCoreHelpers.git
git clone https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git

cd C:\Users\pho\repos\NeuroPy
pip install -e .

cd C:\Users\pho\repos\pyPhoCoreHelpers
pip install -e .

cd C:\Users\pho\repos\pyPhoPlaceCellAnalysis
pip install -e .

## Finally, after installing custom libs, install any extra libs via pip you need:
pip install PyQt5
pip install PyQt6
pip install PyQt6-tools



---
## Creating New Modules

### Solution for Missing Packages after Install:
Even after installing with `pip install -e .` with my correct conda environment active, the module was not visible in Jupyter-lab and wasn't found during imports even when present in an imported project. 

#### Solution:
https://github.com/jupyter/help/issues/342#issuecomment-382837602
This was solved strangely by:
1. installing `nb_conda_kernels` in the ***base*** conda environment (not the one you want to use):
```
mamba install -n base nb_conda_kernels -c conda-forge
```
2. installing `ipykernel` in the conda environment you DO want to use:
```
mamba install ipykernel -c conda-forge
```
3. launching jupyter-lab from the ***base*** environment:
```
conda activate base
jupyter-lab
```
4. finally, in jupyter-lab select the kerenl you want to use from the top right corner. It should now work.



### pyscaffold:
mamba install pyscaffold pyscaffoldext-dsproject tox pre-commit -c conda-forge
https://github.com/pyscaffold/pyscaffold

#### Usage:
adds the `putput` command, to be used like:
`putup --dsproject pyPhoCoreHelpers`

To install, activate your desired conda environment and then run
`pip install -e .`

```
tox -e docs # build package documentation
tox -e build  # to build your package distribution
tox -e publish  # to test your project uploads correctly in test.pypi.org
tox -e publish -- --repository pypi  # to release your package to PyPI
tox -av  # to list all the tasks available
```

See 
https://pyscaffold.org/en/latest/dependencies.html#dependencies
https://pyscaffold.org/projects/dsproject/en/stable/readme.html


```
mamba env create -f environment.yml
conda activate demo-pyphocorehelpers
```
