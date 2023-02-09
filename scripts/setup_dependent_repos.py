""" Clones the dependent repos if they don't exist, and updates them if they do.
TODO: 2023-02-08


git clone

git pull


poetry lock
poetry install
"""

""" Write a python script that does the following for each of the dependent_repos:
dependent_repos = ["../NeuroPy", "../pyPhoCoreHelpers", "../pyPhoPlaceCellAnalysis"]

1. git pull the latest changes from their remotes.
2. Run `poetry lock; poetry install`
"""
import os
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__))) # /home/halechr/repos/Spike3D/scripts
print(f'script_dir: {script_dir}')
root_dir = script_dir.parent

dependent_repos = ["../NeuroPy", "../pyPhoCoreHelpers", "../pyPhoPlaceCellAnalysis"]
dependent_repos_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in dependent_repos]

print(f'{dependent_repos_paths = }')
for repo in dependent_repos_paths:
    print(f'Updating {repo}...')
    if not repo.exists():
        # clone the repo
        os.chdir(repo.parent()) # change directory to the parent of the repo to prepare for cloning
        print(f'\t repo does not exist. Cloning {repo}...')
        os.system(f'git clone {repo}')

    os.chdir(repo)
    os.system("git pull")
    os.system("poetry lock")
    os.system("poetry install") # is this needed? I think it installs in that specific environment.

print(f'done with all.')
os.chdir(root_dir)
