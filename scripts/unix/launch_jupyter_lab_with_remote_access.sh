#!/bin/bash

cd /home/halechr/repos/Spike3D
# ./scripts/unix/repos_pull_changes.sh

# Ensure Poetry is available (optional but recommended)
if ! command -v poetry &> /dev/null
then
    echo "Poetry could not be found. Please install Poetry first."
    exit 1
fi

# Get the path to the Poetry virtual environment
VENV_PATH=$(poetry env info --path)
# VENV_PATH="/home/halechr/Library/VSCode/green/.venv_green"
# /home/halechr/Library/VSCode/green/.venv_green/bin/python
# Source the activate script of the virtual environment
# source $VENV_PATH/bin/activate
# poetry shell

# Run Jupyter Lab within the Poetry environment
poetry run jupyter-lab --no-browser --port=8889 --NotebookApp.ip='0.0.0.0' --NotebookApp.allow_origin='*'
