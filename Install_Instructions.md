## Updated 2023-02-06 Poetry Environment Install


#### Linux (RHEL 8):
```bash
<!-- sudo dnf install wget yum-utils make gcc openssl-devel bzip2-devel libffi-devel zlib-devel
wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tgz  -->
https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_basic_system_settings/assembly_installing-and-using-python_configuring-basic-system-settings
https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/configuring_basic_system_settings/assembly_configuring-the-unversioned-python_configuring-basic-system-settings
# To install Python 3.9.X and set it as the default system python:
sudo yum install python39 python39-pip
sudo alternatives --set python /usr/bin/python3.9
```


#### Great-Lakes:
```bash
module load python/3.9.12
curl -sSL https://install.python-poetry.org | python3 -
```



#### Windows:
##### General Helpers:
See current version of a given command (e.g. `poetry`)
```powershell
Get-Command poetry | Select-Object -ExpandProperty Definition
```


##### Install Chocolatey:
1. Open a Powershell terminal and do the following:

```
# Install `scoop`
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# Install pipx via scoop
scoop install pipx
pipx ensurepath

# Install poetry via pipx
pipx install poetry
pipx inject poetry poetry-plugin-shell

```
2. Check that the installed poetry is the one being referenced
```
Get-Command poetry | Select-Object -ExpandProperty Definition
```
3. #TODO 2025-04-07 17:06: - [ ] Fix



--------------------------- Pre 2025-04-07 ---------------------------

1 (alt-OLD). Open a CMD.exe Administrator Instance and paste the following:
```bash
@REM Install Chocolatey
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

@REM Use Chocolatey to install required system utilities
choco install vscode git curl pyenv-win -y # installs vscode, git, and pyenv-win

@REM Install Python 3.9.13
refreshEnv
pyenv install 3.9.13
pyenv shell 3.9.13
pyenv local 3.9.13 @REM this sets the new install to be the local Python version (in the directory where this command is ran)




pipx install poetry
pipx inject poetry poetry-plugin-shell


@REM Install Poetry
curl -sSL https://install.python-poetry.org | python3 - @REM get poetry and install it

@REM add Poetry to path for current shell only
set PATH=%PATH%;%APPDATA%\Python\Scripts
poetry env use C:\Users\pho\.pyenv\pyenv-win\versions\3.9.13\python.exe
poetry shell

```


#### macOS:
brew install python3 poetry
curl -sSL https://install.python-poetry.org | python3 -




### Install Poetry:
curl -sSL https://install.python-poetry.org | python3 -


poetry add git+https://github.com/CommanderPho/NeuroPy.git
<!-- poetry add git+https://github.com/CommanderPho/pyPhoCoreHelpers.git -->
poetry add git+https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git
poetry add git+https://github.com/CommanderPho/mpl-multitab.git
poetry add git+https://github.com/CommanderPho/ansi2html.git
poetry add git+https://github.com/CommanderPho/vedo.git#release/pho-working
poetry add git+https://github.com/CommanderPho/pyqode.python.git

poetry add git+https://github.com/CommanderPho/cnn_ripple.git


poetry shell


poetry install --all-extras




----
(spike3d-py3.10) (base) ➜  Spike3D git:(master) ✗ which python
/Users/pho/Library/Caches/pypoetry/virtualenvs/spike3d-b38JC3r7-py3.10/bin/python




## Run Poetry environment in globally installed Jupyter-lab (as kernel):
`poetry run ipython kernel install --user --name=spike3d-poetry`
```
(spike3d-py3.9) C:\Users\pho\repos\PhoPy3DPositionAnalysis2021>poetry run ipython kernel install --user --name=spike3d-poetry
Installed kernelspec spike3d-poetry in C:\Users\pho\AppData\Roaming\jupyter\kernels\spike3d-poetry
```



----------------

conda install -c conda-forge jupyterlab
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
git clone https://github.com/CommanderPho/cnn_ripple.git


cd C:\Users\pho\repos\NeuroPy
python -m pip install -e .

cd C:\Users\pho\repos\pyPhoCoreHelpers
python -m pip install -e .

cd C:\Users\pho\repos\pyPhoPlaceCellAnalysis
python -m pip install -e .

cd C:\Users\pho\repos\cnn_ripple
python -m pip install -e .




# Concise version:
python -m pip install -e C:\Users\pho\repos\NeuroPy
python -m pip install -e C:\Users\pho\repos\pyPhoCoreHelpers
python -m pip install -e C:\Users\pho\repos\pyPhoPlaceCellAnalysis
python -m pip install -e C:\Users\pho\repos\cnn_ripple


python -m pip install -e C:\Users\pho\repos\ExternalTesting\pyqode.python

## Non-Editable Variants:
python -m pip install git+https://github.com/CommanderPho/NeuroPy.git
python -m pip install git+https://github.com/CommanderPho/pyPhoCoreHelpers.git
python -m pip install git+https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git
python -m pip install git+https://github.com/CommanderPho/cnn_ripple.git

python -m pip install git+https://github.com/CommanderPho/pyqode.python.git
# TODO - add other package URLS


## Finally, after installing custom libs, install any extra libs via pip you need:
pip install tensorflow findpeaks~=2.4.3 opencv-python
pip install PyQt5
pip install PyQt6-tools



python -m pip install -e C:\Users\pho\repos\NeuroPy
python -m pip install -e C:\Users\pho\repos\pyPhoCoreHelpers
python -m pip install -e C:\Users\pho\repos\pyPhoPlaceCellAnalysis


# Install to new .venv_new:
.venv_new\Scripts\python -m pip install -e C:\Users\pho\repos\Spike3DWorkEnv\NeuroPy
.venv_new\Scripts\python -m pip install -e C:\Users\pho\repos\Spike3DWorkEnv\pyPhoCoreHelpers
.venv_new\Scripts\python -m pip install -e C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis


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


# 2022-07-05 New Packages:
	python -m pip install findpeaks~=2.4.3 opencv-python indexed~=1.2.1 pybursts~=0.1.1 PyQt5Singleton pyqt-checkbox-table-widget tensorflow python-benedict

mamba install tox jupyter-lab cython ipykernel pyvistaqt -c conda-forge


## 2022-11-04 - To Upgrade

mamba env update -n <your-env> --file environment.yml


mamba env update -n mamba_ultimate --file "C:\Users\pho\Desktop\Anaconda Environments Full Backup 2022-11-04\pho_ultimate.yaml"



## #TODO 2025-03-08 21:47: - [ ] Custom Jupyter Kernel to Spike3D Poetry VEnv 

```
(spike3d-py3.9) PS C:\Users\pho\repos\Spike3DWorkEnv\Spike3D> ipython kernel install --user --name=spike3d-global-poetry
Installed kernelspec spike3d-global-poetry in C:\Users\pho\AppData\Roaming\jupyter\kernels\spike3d-global-poetry


ipython kernel install --user --name=spike3d-global-poetry
```
