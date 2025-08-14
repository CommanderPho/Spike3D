#!/bin/bash

# Exports a requirements.txt suitable for use by pip/virtualenv from each repo's poetry project:

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

& poetry run python ../helpers/export_subrepos.py
