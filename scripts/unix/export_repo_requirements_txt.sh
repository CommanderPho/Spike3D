#!/bin/bash

# Exports a requirements.txt suitable for use by pip/virtualenv from each repo's poetry project:

# Define directories
spike3DDirectory=$(dirname $(dirname "$BASH_SOURCE"))
repoParentDirectory=$(dirname "$spike3DDirectory")

directories=(
    "$repoParentDirectory/NeuroPy"
    "$repoParentDirectory/pyPhoCoreHelpers"
    "$repoParentDirectory/pyPhoPlaceCellAnalysis"
    "$spike3DDirectory"
)

# Commands to run in each directory
commands=(
    "poetry export --without-hashes --format=requirements.txt > requirements.txt"
    "poetry export --without-hashes --format=requirements.txt > requirements.txt"
    "poetry export --without-hashes --format=requirements.txt > requirements.txt"
    "poetry export --without-hashes --format=requirements.txt > requirements.txt"
)

# Function to export requirements
export_requirements_from_poetry() {
    echo "Exporting requirements.txt from poetry repos..."
    for i in "${!directories[@]}"; do
        directory="${directories[i]}"
        command="${commands[i]}"
        echo "Running in $directory"
        cd "$directory" || exit
        eval "$command"
        echo "done."
    done
}

export_requirements_from_poetry
