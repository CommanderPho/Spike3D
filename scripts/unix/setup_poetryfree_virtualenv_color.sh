#!/bin/bash

# Prompt for a color name if not supplied as the first argument
colorFolderColorName="$1"
if [ -z "$colorFolderColorName" ]; then
    read -p "Enter a color name (like 'yellow') without quotes: " colorFolderColorName
fi

echo "Creating a new virtual-env with colorFolderColorName: $colorFolderColorName"

envName=".venv_$colorFolderColorName"
fullEnvParentPath="$HOME/Library/VSCode/$colorFolderColorName"
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
pyenv local 3.9.13
pyenv exec virtualenv "$envName"
deactivate 2>/dev/null || true
source "$fullActivateScriptPath"

spike3d_repo_path="$HOME/repos/Spike3D"

echo "Using spike3d_repo_path: $spike3d_repo_path..."

# Check if the Spike3D repository exists
if [ ! -d "$spike3d_repo_path" ]; then
    echo "The directory $spike3d_repo_path does not exist. You will need to finish setup by symlinking and registering the kernel"
else
    "$fullPythonPath" -m ensurepip --default-pip
    "$fullPythonPath" -m pip install --upgrade pip

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
