#!/bin/bash

# xfce4-terminal --initial-title='Neuropy' --working-directory='/home/halechr/repos/NeuroPy' --tab 
# xfce4-terminal --initial-title='pyPhoCoreHelpers' --working-directory='/home/halechr/repos/pyPhoCoreHelpers'
# xfce4-terminal --initial-title='pyPhoPlaceCellAnalysis' --working-directory='/home/halechr/repos/pyPhoPlaceCellAnalysis'
# xfce4-terminal --initial-title='Spike3D' --working-directory='/home/halechr/repos/Spike3D'

# xfce4-terminal --title='Neuropy' --working-directory='/home/halechr/repos/NeuroPy' --tab 
# xfce4-terminal --title='pyPhoCoreHelpers' --working-directory='/home/halechr/repos/pyPhoCoreHelpers'
# xfce4-terminal --title='pyPhoPlaceCellAnalysis' --working-directory='/home/halechr/repos/pyPhoPlaceCellAnalysis'
# xfce4-terminal --title='Spike3D' --working-directory='/home/halechr/repos/Spike3D'

## Launches all tabs in a single terminal:
xfce4-terminal --window --title='Neuropy' --working-directory='/home/halechr/repos/NeuroPy' --tab --title='pyPhoCoreHelpers' --working-directory='/home/halechr/repos/pyPhoCoreHelpers' --tab --title='pyPhoPlaceCellAnalysis' --working-directory='/home/halechr/repos/pyPhoPlaceCellAnalysis' --tab --title='Spike3D' --working-directory='/home/halechr/repos/Spike3D'

