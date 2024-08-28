#!/bin/bash

## Moves the individual poetry .venvs to the turbo/Pho directory on Greatlakes


# cp -R '/home/halechr/repos/NeuroPy/.venv' '/home/halechr/cloud/turbo/Pho/Environments/NeuroPy_venv'

# cp -R '/home/halechr/repos/NeuroPy/.venv' '/home/halechr/cloud/turbo/Pho/Environments/NeuroPy_venv'
# rm -rf '/home/halechr/repos/NeuroPy/.venv'

mv '/home/halechr/repos/NeuroPy/.venv' '/home/halechr/cloud/turbo/Pho/Environments/NeuroPy_venv'
ln -s '/home/halechr/cloud/turbo/Pho/Environments/NeuroPy_venv' '/home/halechr/repos/NeuroPy/.venv'


mv '/home/halechr/repos/pyPhoCoreHelpers/.venv' '/home/halechr/cloud/turbo/Pho/Environments/pyPhoCoreHelpers_venv'
ln -s '/home/halechr/cloud/turbo/Pho/Environments/pyPhoCoreHelpers_venv' '/home/halechr/repos/pyPhoCoreHelpers/.venv'

# poetry env use '/home/halechr/cloud/turbo/Pho/Environments/pyPhoCoreHelpers_venv/bin/python' # NOT NEEDED

mv '/home/halechr/repos/pyPhoPlaceCellAnalysis/.venv' '/home/halechr/cloud/turbo/Pho/Environments/pyPhoPlaceCellAnalysis_venv'
ln -s '/home/halechr/cloud/turbo/Pho/Environments/pyPhoPlaceCellAnalysis_venv' '/home/halechr/repos/pyPhoPlaceCellAnalysis/.venv'


mv '/home/halechr/repos/Spike3D/.venv' '/home/halechr/repos/Spike3D/Spike3D_venv'


mv '/home/halechr/repos/Spike3D/Spike3D_venv' '/home/halechr/cloud/turbo/Pho/Environments/Spike3D_venv'


mv '/home/halechr/repos/Spike3D/.venv' '/home/halechr/cloud/turbo/Pho/Environments/Spike3D_venv'
ln -s '/home/halechr/cloud/turbo/Pho/Environments/Spike3D_venv' '/home/halechr/repos/Spike3D/.venv'


# ```bash
# bash-4.4$ poetry env list --full-path
# /home/halechr/repos/pyPhoCoreHelpers/.venv (Activated)
# bash-4.4$ poetry env info

# Virtualenv
# Python:         3.9.12
# Implementation: CPython
# Path:           /home/halechr/repos/pyPhoCoreHelpers/.venv
# Executable:     /home/halechr/repos/pyPhoCoreHelpers/.venv/bin/python
# Valid:          True

# System
# Platform:   linux
# OS:         posix
# Python:     3.9.12
# Path:       /sw/pkgs/arc/python/3.9.12
# Executable: /sw/pkgs/arc/python/3.9.12/bin/python3.9

# ```
# /home/halechr/repos/Spike3D/.venv

# ```bash
# bash-4.4$ poetry env list --full-path
# /home/halechr/repos/Spike3D/.venv (Activated)
# bash-4.4$ poetry env info

# Virtualenv
# Python:         3.9.12
# Implementation: CPython
# Path:           /home/halechr/repos/Spike3D/.venv
# Executable:     /home/halechr/repos/Spike3D/.venv/bin/python
# Valid:          True

# System
# Platform:   linux
# OS:         posix
# Python:     3.9.12
# Path:       /sw/pkgs/arc/python/3.9.12
# Executable: /sw/pkgs/arc/python/3.9.12/bin/python3.9
# ```