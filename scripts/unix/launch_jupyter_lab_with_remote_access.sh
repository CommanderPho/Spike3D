#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
PROJECT_DIR="/home/halechr/repos/Spike3D"
SCRIPT_PULL="/home/halechr/repos/Spike3D/scripts/unix/repos_pull_changes.sh"
JUPYTER_PORT=8889
JUPYTER_LOG="jupyter.log"

# Function to copy text to clipboard using xclip
copy_to_clipboard() {
    local text="$1"
    echo -n "$text" | xclip -selection clipboard
}

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

# Start Jupyter Lab in the background, redirecting output to a log file
echo "Starting Jupyter Lab..."
poetry run jupyter-lab --no-browser --port="$JUPYTER_PORT" --NotebookApp.ip='0.0.0.0' --NotebookApp.allow_origin='*' > "$JUPYTER_LOG" 2>&1 &

# Get the PID of the Jupyter Lab process
JUPYTER_PID=$!

# Function to gracefully terminate Jupyter Lab on script exit
cleanup() {
    echo "Shutting down Jupyter Lab..."
    kill "$JUPYTER_PID"
    wait "$JUPYTER_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait until Jupyter Lab is ready by checking the log file for the URL
echo "Waiting for Jupyter Lab to start..."
URL=""
TIMEOUT=30  # seconds
START_TIME=$(date +%s)

while true; do
    if grep -oP 'http://127\.0\.0\.1:'"$JUPYTER_PORT"'\?token=\w+' "$JUPYTER_LOG" > /dev/null; then
        URL=$(grep -oP 'http://127\.0\.0\.1:'"$JUPYTER_PORT"'\?token=\w+' "$JUPYTER_LOG" | head -n1)
        break
    fi

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Timed out waiting for Jupyter Lab to start."
        exit 1
    fi

    sleep 1
done

# Copy the URL to the clipboard
if [ -n "$URL" ]; then
    echo "Jupyter Lab is running at: $URL"
    copy_to_clipboard "$URL" && echo "Jupyter URL copied to clipboard."
else
    echo "Failed to retrieve Jupyter Lab URL."
    exit 1
fi

# Optionally, open the URL in the default browser
# Uncomment the following line if you want to automatically open the browser
# xdg-open "$URL"

# Keep the script running to maintain Jupyter Lab
wait "$JUPYTER_PID"