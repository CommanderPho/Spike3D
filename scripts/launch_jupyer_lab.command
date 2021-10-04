#!/bin/bash
# Launches the jupyer notebook
# PATH="/path/to/your/miniconda3/bin:<rest of your path>"

eval "$(conda shell.bash hook)"
conda activate viz3d
cd ~ && python -m jupyter lab --port 8888 --no-browser &
open -a "Google Chrome" http://127.0.0.1:8888/lab
