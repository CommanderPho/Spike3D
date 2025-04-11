#!/bin/bash

# Exports typestubs from each repo:

# Define directories from the path to the current script. Needs to change relative to the scripts location within the scripts folder.
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
script_dir=$(dirname $(dirname "$script_path"))
spike3DDirectory=$(dirname "$script_dir")
repoParentDirectory=$(dirname "$spike3DDirectory") # repoParentDirectory='$HOME/repos/'

# Log the key directory paths
echo "--------- Directory Paths ---------"
echo "Script path: $script_path"
echo "Script directory: $script_dir"
echo "Spike3D directory: $spike3DDirectory"
echo "Repository parent directory: $repoParentDirectory"
echo "---------------------------------"

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
