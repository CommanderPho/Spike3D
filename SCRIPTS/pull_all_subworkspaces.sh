#!/bin/bash
# Fetches latest updates for Spike3D and its four main dependent repos.
# Repos are expected at sibling paths (../NeuroPy, ../pyPhoCoreHelpers, ../pyPhoPlaceCellAnalysis).
# If not found, they are cloned from GitHub into a fallback directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPIKE3D_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SIBLING_DIR="$(cd "$SPIKE3D_DIR/.." && pwd)"

FALLBACK_DIR="$HOME/repos"

REPOS=(
    "NeuroPy:https://github.com/CommanderPho/NeuroPy.git"
    "pyPhoCoreHelpers:https://github.com/CommanderPho/pyPhoCoreHelpers.git"
    "pyPhoPlaceCellAnalysis:https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git"
)

pull_repo() {
    local repo_path="$1"
    local repo_name="$2"
    local repo_url="$3"

    if [ -d "$repo_path" ]; then
        echo "============= $repo_name (pull):"
        cd "$repo_path" && git pull
    else
        echo "============= $repo_name (clone into fallback $FALLBACK_DIR):"
        mkdir -p "$FALLBACK_DIR"
        git clone "$repo_url" "$FALLBACK_DIR/$repo_name"
    fi
}

echo "============= Spike3D:"
cd "$SPIKE3D_DIR" && git pull

for entry in "${REPOS[@]}"; do
    repo_name="${entry%%:*}"
    repo_url="${entry#*:}"
    sibling_path="$SIBLING_DIR/$repo_name"
    fallback_path="$FALLBACK_DIR/$repo_name"

    if [ -d "$sibling_path" ]; then
        pull_repo "$sibling_path" "$repo_name" "$repo_url"
    elif [ -d "$fallback_path" ]; then
        pull_repo "$fallback_path" "$repo_name" "$repo_url"
    else
        echo "============= $repo_name (clone into fallback $FALLBACK_DIR):"
        mkdir -p "$FALLBACK_DIR"
        git clone "$repo_url" "$fallback_path"
    fi
done

echo "Done with all."
