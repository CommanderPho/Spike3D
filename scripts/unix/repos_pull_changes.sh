#!/bin/bash

# Get the script's directory and calculate project root and parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE3D_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$(cd "$SPIKE3D_DIR/.." && pwd)"

cd "$PARENT_DIR/NeuroPy/"
echo "============= NeuroPy:"
git pull
cd "$PARENT_DIR/pyPhoCoreHelpers/"
echo "============= pyPhoCoreHelpers:"
git pull
cd "$PARENT_DIR/pyPhoPlaceCellAnalysis/"
echo "============= pyPhoPlaceCellAnalysis:"
git pull
cd "$SPIKE3D_DIR/"
echo "============= Spike3D:"
git pull
echo "done with all."