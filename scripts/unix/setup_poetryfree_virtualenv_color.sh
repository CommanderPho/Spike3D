#!/bin/bash

# Prompt for a color name if not supplied as the first argument
colorFolderColorName="$1"
if [ -z "$colorFolderColorName" ]; then
    read -p "Enter a color name (like 'yellow') without quotes: " colorFolderColorName
fi

echo "Creating a new virtual-env with colorFolderColorName: $colorFolderColorName"

# env_create_parent="$HOME/Library"
env_create_parent="."
# env_create_parent="/scratch/kdiba_root/kdiba1/halechr/Library" # scratch disk
# /tmpssd/

echo "env_create_parent: $env_create_parent"
mkdir -p "$env_create_parent"


SCRIPT_FILE_PARENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SPIKE3D_DIR="$( cd "$SCRIPT_FILE_PARENT_DIR/../.." && pwd )"
echo "SCRIPT_FILE_PARENT_DIR: $SCRIPT_FILE_PARENT_DIR"
echo "SPIKE3D_DIR: $SPIKE3D_DIR"

envName=".venv_$colorFolderColorName"
fullEnvParentPath="$env_create_parent/$colorFolderColorName"
fullEnvPath="$fullEnvParentPath/$envName"
fullActivateScriptPath="$fullEnvPath/bin/activate"
fullPythonPath="$fullEnvPath/bin/python"

echo "fullEnvParentPath: $fullEnvParentPath"
echo "fullEnvPath: $fullEnvPath"
echo "fullActivateScriptPath: $fullActivateScriptPath"
echo "fullPythonPath: $fullPythonPath"

# Create the directory if it doesn't exist
mkdir -p "$fullEnvParentPath"
cd "$fullEnvParentPath"

# Use pyenv to set local Python version and create a virtual environment
# pyenv local 3.9.19
# pyenv exec virtualenv "$envName"
## requires pyenv, not present on GL
pyenv local 3.9.13
# pyenv exec virtualenv $envName

deactivate
python -m virtualenv "$envName"
# deactivate 2>/dev/null || true
deactivate
source "$fullActivateScriptPath"

spike3d_repo_path="$HOME/repos/Spike3D"

echo "Using spike3d_repo_path: $spike3d_repo_path..."

# Check if the Spike3D repository exists
if [ ! -d "$spike3d_repo_path" ]; then
    echo "The directory $spike3d_repo_path does not exist. You will need to finish setup by symlinking and registering the kernel"
else
    "$fullPythonPath" -m ensurepip --default-pip
    "$fullPythonPath" -m pip install --upgrade pip
    "$fullPythonPath" -m pip install setuptools wheel

    ## Add build dependencies
    external_dependent_repos=("../pyqode.core" "../pyqode.python") #  "../silx"
    for repo in "${external_dependent_repos[@]}"; do
        if [ -d "$repo" ]; then
            cd "$repo"
            python setup.py sdist bdist_wheel --dist-dir=./dist/
            cd dist/
            whl_files=($(ls *.whl 2>/dev/null | grep -v "current.whl"))
            [ ${#whl_files[@]} -gt 0 ] && { [ -L "current.whl" ] && rm -f current.whl; ln -s "${whl_files[-1]}" current.whl; }
            cd ../
        else
            echo "Warning: Directory $repo not found"
        fi
    done

    # Install requirements and packages from Spike3DWorkEnv
    "$fullPythonPath" -m pip install -r "$HOME/repos/Spike3D/requirements.txt"
    "$fullPythonPath" -m pip install -e "$HOME/repos/NeuroPy"
    "$fullPythonPath" -m pip install -e "$HOME/repos/pyPhoCoreHelpers"
    "$fullPythonPath" -m pip install -e "$HOME/repos/pyPhoPlaceCellAnalysis"

    # Link the new env as a symlink
    new_env_spike3D_folder_target="$spike3d_repo_path/$envName"
    echo "$new_env_spike3D_folder_target"
    ln -s -f "$fullEnvPath" "$new_env_spike3D_folder_target"
    cd "$new_env_spike3D_folder_target"

    # Install the IPython kernel
    python -m ipykernel install --user --name="spike3d-$colorFolderColorName"

    # Add to .gitignore
    echo "$envName" >> "$spike3d_repo_path/.gitignore"
    echo "Completed successfully! done."
fi

# fullPythonPath="$HOME/Library/VSCode/green/.venv_green/bin/python"
# '$HOME/Library/VSCode/green/.venv_green/bin/python'
# '$HOME/Library/VSCode/green/.venv_green/bin/activate'
# source $HOME/Library/VSCode/green/.venv_green/bin/activate