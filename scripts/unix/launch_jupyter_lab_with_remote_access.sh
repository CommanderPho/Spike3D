#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
PROJECT_DIR="/home/halechr/repos/Spike3D"
SCRIPT_PULL="/home/halechr/repos/Spike3D/scripts/unix/repos_pull_changes.sh"
JUPYTER_PORT=8889
JUPYTER_LOG="$PROJECT_DIR/jupyter.log"  # Absolute path
CLIPBOARD_CMD="clip"  # Assuming 'clip' alias is set to xclip
TIMEOUT=60  # seconds to wait for Jupyter to start

# Function to copy text to clipboard
copy_to_clipboard() {
    local text="$1"
    echo -n "$text" | $CLIPBOARD_CMD
}

# Navigate to your project directory
echo "Navigating to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "Failed to navigate to $PROJECT_DIR"; exit 1; }

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
JUPYTER_PID=$!
echo "Jupyter Lab PID: $JUPYTER_PID"

# Function to gracefully terminate Jupyter Lab on script exit
cleanup() {
    echo "Shutting down Jupyter Lab (PID: $JUPYTER_PID)..."
    kill "$JUPYTER_PID"
    wait "$JUPYTER_PID" 2>/dev/null
}
trap cleanup EXIT

# Wait until Jupyter Lab is ready by checking the log file for the URL
echo "Waiting for Jupyter Lab to start and generate the URL..."
URL=""
START_TIME=$(date +%s)

while true; do
    if [ -f "$JUPYTER_LOG" ]; then
        # Attempt to extract the URL
        URL=$(grep -oP 'http://127\.0\.0\.1:'"$JUPYTER_PORT"'/lab\?token=\w+' "$JUPYTER_LOG" | head -n1)
        if [ -n "$URL" ]; then
            echo "Jupyter Lab URL found: $URL"
            break
        fi
    fi

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Timed out waiting for Jupyter Lab to start."
        echo "Checking Jupyter Log for errors:"
        cat "$JUPYTER_LOG"
        exit 1
    fi

    sleep 2
done

# Copy the URL to the clipboard
if [ -n "$URL" ]; then
    echo "Copying Jupyter URL to clipboard..."
    copy_to_clipboard "$URL" && echo "Jupyter URL copied to clipboard."
else
    echo "Failed to retrieve Jupyter Lab URL."
    exit 1
fi

# Optionally, open the URL in the default browser
# Uncomment the following line if you want to automatically open the browser
# xdg-open "$URL"

echo "Jupyter Lab is running. Access it at: $URL"

# Keep the script running to maintain Jupyter Lab
wait "$JUPYTER_PID"