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

enable_install_for_child_repos = False

script_dir = Path(os.path.dirname(os.path.abspath(__file__))) # /home/halechr/repos/Spike3D/scripts
print(f'script_dir: {script_dir}')
root_dir = script_dir.parent # Spike3D root repo dir
os.chdir(root_dir)
os.system("git pull") # pull changes first for main repo

## Pull for children repos:
dependent_repos = ["../NeuroPy", "../pyPhoCoreHelpers", "../pyPhoPlaceCellAnalysis"]
dependent_repos_urls = ["https://github.com/CommanderPho/NeuroPy.git", "https://github.com/CommanderPho/pyPhoCoreHelpers.git", "https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git"]
dependent_repos_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in dependent_repos]
poetry_repo_tuples = list(zip(dependent_repos_paths, dependent_repos_urls, [False]*len(dependent_repos_paths)))


external_dependent_repos = ["../pyqode.core", "../pyqode.python"]
external_dependent_binary_repo_urls = ["https://github.com/CommanderPho/pyqode.core.git", "https://github.com/CommanderPho/pyqode.python.git"]
external_dependent_repo_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in external_dependent_repos]
binary_repo_tuples = list(zip(external_dependent_repo_paths, external_dependent_binary_repo_urls, [True]*len(external_dependent_repo_paths)))



def setup_repo(repo_path, repo_url, is_binary_repo=False):
    if not repo_path.exists():
        # clone the repo
        os.chdir(repo_path.parent) # change directory to the parent of the repo to prepare for cloning
        print(f'\t repo does not exist. Cloning {repo_path}...')
        os.system(f'git clone {repo_url}')

    os.chdir(repo_path)
    os.system("git pull")

    if is_binary_repo:
        ## For binary repos:
        os.system("python setup.py sdist bdist_wheel")
    else:
        # For poetry repos
        os.system("poetry lock")
        if enable_install_for_child_repos:
            os.system("poetry install") # is this needed? I think it installs in that specific environment.



print(f'{dependent_repos_paths = }')
# for i, repo_path in enumerate(dependent_repos_paths):

for i, (repo_path, repo_url, is_binary_repo) in enumerate(poetry_repo_tuples + binary_repo_tuples):
    print(f'Updating {repo_path}...')
    setup_repo(repo_path, repo_url, is_binary_repo=is_binary_repo)

    # if not repo.exists():
    #     # clone the repo
    #     os.chdir(repo.parent) # change directory to the parent of the repo to prepare for cloning
    #     print(f'\t repo does not exist. Cloning {repo}...')
    #     os.system(f'git clone {dependent_repos_urls[i]}')

    # os.chdir(repo)
    # os.system("git pull")
    # os.system("poetry lock")
    # if enable_install_for_child_repos:
    #     os.system("poetry install") # is this needed? I think it installs in that specific environment.

os.chdir(root_dir) # change back to the root repo dir
os.system("poetry lock")
os.system("poetry install --all-extras") # is this needed? I think it installs in that specific environment.
print(f'done with all.')

os.system("poetry run ipython kernel install --user --name=spike3d-poetry") # run this to install the kernel for the poetry environment

