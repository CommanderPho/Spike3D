#!/bin/bash
# Get stash comment from first argument, default to "GL" if not provided
stash_comment="${1:-GL}"
cd ~/repos/NeuroPy/
git stash save "$stash_comment"
cd ~/repos/pyPhoCoreHelpers/
git stash save "$stash_comment"
cd ~/repos/pyPhoPlaceCellAnalysis/
git stash save "$stash_comment"
cd ~/repos/Spike3D/
git stash save "$stash_comment"