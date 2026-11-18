#!/bin/bash

# Commands to run in each repo
cmd_git_ops='git pull; git status; exec bash'

# Geometry for the first terminal (e.g., 130 columns wide by 30 lines tall)
geometry1="900x30+0+0"

# Launch all tabs in a single terminal and run the commands
xfce4-terminal \
--window \
--geometry=$geometry1 \
--tab --title='Spike3D .venv' --working-directory='$HOME/repos/Spike3D' --command="bash -c 'source .venv/bin/activate; exec bash'" &

Launch Visual Studio Code separately
$HOME/bin/VSCode-linux-x64/bin/code &

