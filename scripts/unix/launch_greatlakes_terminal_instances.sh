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

# Launch Visual Studio Code separately
/home/halechr/bin/VSCode-linux-x64/bin/code &

# Launch a second xfce4-terminal window and execute the JupyterLab launch script
xfce4-terminal \
--window \
--title='Jupyter Lab' \
--working-directory='/home/halechr/repos/Spike3D' \
--command="/home/halechr/repos/Spike3D/scripts/unix/launch_jupyter_lab_with_remote_access.sh" &
