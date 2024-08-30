#!/bin/bash

# Exports typestubs from each repo:

# Define directories
# spike3DDirectory=$(dirname $(dirname "$BASH_SOURCE"))
spike3DDirectory='/home/halechr/repos/Spike3D'

# repoParentDirectory='/home/halechr/repos/'
repoParentDirectory=$(dirname "$spike3DDirectory")

directories=(
    "$repoParentDirectory/NeuroPy"
    "$repoParentDirectory/pyPhoCoreHelpers"
    "$repoParentDirectory/pyPhoPlaceCellAnalysis"
    "$spike3DDirectory"
)

# Commands to run in each directory
commands=(
    "pyright --createstub neuropy"
    "pyright --createstub pyphocorehelpers"
    "pyright --createstub pyphoplacecellanalysis"
    "poetry shell && pyright --createstub neuropy && pyright --createstub pyphocorehelpers && pyright --createstub pyphoplacecellanalysis"
)

# Function to export typestubs
export_pyright_typestubs() {
    echo "Exporting typestubs using PyRight..."
    for i in "${!directories[@]}"; do
        directory="${directories[i]}"
        command="${commands[i]}"
        echo "Running in $directory"
        cd "$directory" || exit
        rm -rf "$directory/typings"
        eval "$command"
        echo "done."
    done
}

export_pyright_typestubs
