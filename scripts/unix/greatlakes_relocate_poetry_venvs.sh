#!/bin/bash

## Moves the individual poetry .venvs to the turbo/Pho directory on Greatlakes


# cp -R '$HOME/repos/NeuroPy/.venv' '$HOME/cloud/turbo/Pho/Environments/NeuroPy_venv'

# cp -R '$HOME/repos/NeuroPy/.venv' '$HOME/cloud/turbo/Pho/Environments/NeuroPy_venv'
# rm -rf '$HOME/repos/NeuroPy/.venv'

mv '$HOME/repos/NeuroPy/.venv' '$HOME/cloud/turbo/Pho/Environments/NeuroPy_venv'
ln -s '$HOME/cloud/turbo/Pho/Environments/NeuroPy_venv' '$HOME/repos/NeuroPy/.venv'


mv '$HOME/repos/pyPhoCoreHelpers/.venv' '$HOME/cloud/turbo/Pho/Environments/pyPhoCoreHelpers_venv'
ln -s '$HOME/cloud/turbo/Pho/Environments/pyPhoCoreHelpers_venv' '$HOME/repos/pyPhoCoreHelpers/.venv'

# poetry env use '$HOME/cloud/turbo/Pho/Environments/pyPhoCoreHelpers_venv/bin/python' # NOT NEEDED

mv '$HOME/repos/pyPhoPlaceCellAnalysis/.venv' '$HOME/cloud/turbo/Pho/Environments/pyPhoPlaceCellAnalysis_venv'
ln -s '$HOME/cloud/turbo/Pho/Environments/pyPhoPlaceCellAnalysis_venv' '$HOME/repos/pyPhoPlaceCellAnalysis/.venv'


mv '$HOME/repos/Spike3D/.venv' '$HOME/repos/Spike3D/Spike3D_venv'


mv '$HOME/repos/Spike3D/Spike3D_venv' '$HOME/cloud/turbo/Pho/Environments/Spike3D_venv'


mv '$HOME/repos/Spike3D/.venv' '$HOME/cloud/turbo/Pho/Environments/Spike3D_venv'
ln -s '$HOME/cloud/turbo/Pho/Environments/Spike3D_venv' '$HOME/repos/Spike3D/.venv'


# ```bash
# bash-4.4$ poetry env list --full-path
# $HOME/repos/pyPhoCoreHelpers/.venv (Activated)
# bash-4.4$ poetry env info

# Virtualenv
# Python:         3.9.12
# Implementation: CPython
# Path:           $HOME/repos/pyPhoCoreHelpers/.venv
# Executable:     $HOME/repos/pyPhoCoreHelpers/.venv/bin/python
# Valid:          True

# System
# Platform:   linux
# OS:         posix
# Python:     3.9.12
# Path:       /sw/pkgs/arc/python/3.9.12
# Executable: /sw/pkgs/arc/python/3.9.12/bin/python3.9

# ```
# $HOME/repos/Spike3D/.venv

# ```bash
# bash-4.4$ poetry env list --full-path
# $HOME/repos/Spike3D/.venv (Activated)
# bash-4.4$ poetry env info

# Virtualenv
# Python:         3.9.12
# Implementation: CPython
# Path:           $HOME/repos/Spike3D/.venv
# Executable:     $HOME/repos/Spike3D/.venv/bin/python
# Valid:          True

# System
# Platform:   linux
# OS:         posix
# Python:     3.9.12
# Path:       /sw/pkgs/arc/python/3.9.12
# Executable: /sw/pkgs/arc/python/3.9.12/bin/python3.9
# ```