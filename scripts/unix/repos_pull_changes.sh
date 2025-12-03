#!/bin/bash
cd ~/repos/NeuroPy/
echo "============= NeuroPy:"
git pull
cd ~/repos/pyPhoCoreHelpers/
echo "============= pyPhoCoreHelpers:"
git pull
cd ~/repos/pyPhoPlaceCellAnalysis/
echo "============= pyPhoPlaceCellAnalysis:"
git pull
cd ~/repos/Spike3D/
echo "============= Spike3D:"
git pull
echo "done with all."