#!/bin/bash

# Commands to run in each repo
cmd_git_ops='git pull; git status; exec bash'

# Launch all tabs in a single terminal and run the commands
xfce4-terminal \
--window \
--tab --title='Neuropy' --working-directory='/home/halechr/repos/NeuroPy' --command="bash -c '$cmd_git_ops'" \
--tab --title='pyPhoCoreHelpers' --working-directory='/home/halechr/repos/pyPhoCoreHelpers' --command="bash -c '$cmd_git_ops'" \
--tab --title='pyPhoPlaceCellAnalysis' --working-directory='/home/halechr/repos/pyPhoPlaceCellAnalysis' --command="bash -c '$cmd_git_ops'" \
--tab --title='Spike3D' --working-directory='/home/halechr/repos/Spike3D' --command="bash -c '$cmd_git_ops'" &

# Ensure the JupyterLab window opens immediately
sleep 1

# Launch Visual Studio Code separately
/home/halechr/bin/VSCode-linux-x64/bin/code &

# Wait a bit to make sure the VS Code command does not interfere with the terminal launch
sleep 1

# Launch a second xfce4-terminal window and execute the JupyterLab launch script
xfce4-terminal \
--window \
--title='Jupyter Lab' \
--working-directory='/home/halechr/repos' \
--command="bash /home/halechr/repos/scripts/unix/launch_jupyter_lab_with_remote_access.sh" &

