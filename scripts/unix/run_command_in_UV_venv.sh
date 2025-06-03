#!/bin/bash

# Script to activate UV virtual environment and run a command on a supercomputer
# Usage: ./run_uv_command.sh [command_to_run]
# Example: ./run_uv_command.sh "ipython ProcessBatchOutputs_GENERIC.ipy --qclu=1,2,4,6,7,8,9 --fr_thresh=5.0"

# Default values
REPO_PATH="$HOME/repos/Spike3D"
UV_ENV_PATH=".UV_venv"
DEFAULT_COMMAND="ipython"

# Parse command line arguments
COMMAND_TO_RUN="${1:-$DEFAULT_COMMAND}"

# Echo information about what we're doing
echo "Working directory: $REPO_PATH"
echo "UV environment: $UV_ENV_PATH"
echo "Command to run: $COMMAND_TO_RUN"

# Change to the repository directory
cd "$REPO_PATH" || { echo "Failed to change to directory $REPO_PATH"; exit 1; }

# Check if UV environment exists
if [ ! -d "$UV_ENV_PATH" ]; then
    echo "UV environment not found at $UV_ENV_PATH"
    echo "Creating new UV environment..."
    uv venv "$UV_ENV_PATH"
fi

# Activate the UV environment and run the command
echo "Activating UV environment and running command..."
source "$UV_ENV_PATH/bin/activate" && eval "$COMMAND_TO_RUN"

# Check if the command executed successfully
if [ $? -eq 0 ]; then
    echo "Command executed successfully"
else
    echo "Command failed with exit code $?"
fi