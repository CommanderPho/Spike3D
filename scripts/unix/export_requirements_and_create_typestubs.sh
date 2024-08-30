#!/bin/bash

# Run both scripts
"$BASH_SOURCE/$(dirname $BASH_SOURCE)/export_repo_requirements_txt.sh"
"$BASH_SOURCE/$(dirname $BASH_SOURCE)/export_repo_typestubs.sh"
