#!/bin/bash

# Commands to run in each repo
cmd_git_ops='git pull; git status; exec bash'

# Geometry for the first terminal (e.g., 130 columns wide by 30 lines tall)
geometry1="1300x30+0+0"

# Launch all tabs in a single terminal and run the commands
xfce4-terminal \
--window \
--geometry=$geometry1 \
--tab --title='Neuropy' --working-directory='$HOME/repos/NeuroPy' --command="bash -c '$cmd_git_ops'" \
--tab --title='pyPhoCoreHelpers' --working-directory='$HOME/repos/pyPhoCoreHelpers' --command="bash -c '$cmd_git_ops'" \
--tab --title='pyPhoPlaceCellAnalysis' --working-directory='$HOME/repos/pyPhoPlaceCellAnalysis' --command="bash -c '$cmd_git_ops'" \
--tab --title='Spike3D' --working-directory='$HOME/repos/Spike3D' --command="bash -c '$cmd_git_ops'" &

# Launch Visual Studio Code separately
$HOME/bin/VSCode-linux-x64/bin/code &

# # Geometry for the Jupyter terminal (e.g., 90 columns wide by 30 lines tall, offset by 100 pixels to the right and down)
# geometry2="90x30+100+100"

# # Launch a second xfce4-terminal window and execute the JupyterLab launch script
# xfce4-terminal \
# --window \
# --geometry=$geometry2 \
# --title='Jupyter Lab' \
# --working-directory='$HOME/repos' \
# --command="bash $HOME/repos/scripts/unix/launch_jupyter_lab_with_remote_access.sh" &