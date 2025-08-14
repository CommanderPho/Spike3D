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
import sys
from pathlib import Path
from datetime import datetime
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))

import glob


from helpers.poetry_helpers import PoetryHelpers, VersionType
from helpers.source_code_helpers import did_file_hash_change # for finding .whl file after building binary repo
from helpers.git_helpers import GitHelpers
from pyphocorehelpers.scripts.export_subrepos import export_poetry_repo

# Get command line input arguments:
parser = argparse.ArgumentParser()
# parser.add_argument('--name', help='the name to greet')
parser.add_argument('--force', action='store_const', help='resets child repos by forcefully pulling from remote', const=True, default=False)
parser.add_argument('--skip_lock', action='store_const', help='whether to skip `poetry lock` for child repos', const=True, default=False)
parser.add_argument('--skip_building_templates', action='store_const', help='whether to skip templating the pyproject.toml file', const=True, default=False)
parser.add_argument('--skip_building_binary_repos', action='store_const', help='whether to skip building child binary repos', const=True, default=False)

group_build_mode = parser.add_mutually_exclusive_group()
group_build_mode.add_argument('--release', action='store_true', help='enable release mode', default=False)
group_build_mode.add_argument('--dev', action='store_true', help='enable development mode', default=True)

args = parser.parse_args()

""" 

--skip_lock 

"""

script_dir = Path(os.path.dirname(os.path.abspath(__file__))) # /home/halechr/repos/Spike3D/scripts
print(f'script_dir: {script_dir}')
root_dir = script_dir.parent # Spike3D root repo dir
os.chdir(root_dir)



# ==================================================================================================================== #
# Repo Processing                                                                                                      #
# ==================================================================================================================== #
def _reset_local_changes(repo_path):
    """ Resets local changes to the repo."""
    os.chdir(repo_path)
    # os.system("git reset --hard HEAD")
    # os.system("git clean -f -d")
    date_now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    os.system(f'git stash save "stash_{date_now_str}"')
    # os.system(f'git stash') # PREV CODE, NO SAVE
    os.system("git stash drop")


def _process_poetry_repo(repo_path, is_release=False, enable_build_pyproject_toml=True, skip_lock=False, enable_install=False):
    ## Build final pyproj.toml file
    if enable_build_pyproject_toml:
        final_pyproject_toml_path = PoetryHelpers.build_pyproject_toml_file(repo_path, is_release=is_release, pyproject_final_file_name = 'pyproject.toml')
    else:
        print(f'skipping build pyproject.toml for {repo_path}')
        final_pyproject_toml_path = 'pyproject.toml'

    ## check if the hash of the pyproject.toml changed. If it did, the new hash is already written.
    did_project_file_change: bool = did_file_hash_change(final_pyproject_toml_path)
    if not skip_lock:
        if did_project_file_change:
            os.system("poetry lock")
            # output_requirements_file_path = Path(repo_path).joinpath('requirements.txt').resolve()
            export_poetry_repo(repo_path, output_file_path='requirements.txt')
            # os.system("poetry export --without-hashes --format=requirements.txt > requirements.txt") # export the requirements for pip once lock is complete
            
        else:
            print(f'\t skipping lock for {repo_path} because project file did not change.')
    else:
        print(f'skipping lock for {repo_path}')

    if enable_install:
        os.system("poetry install") # is this needed? I think it installs in that specific environment.



def _process_binary_repo(repo_path, skip_building=False):
    """ builds binary repos if needed """
    if not skip_building:
        # os.system("pyenv local 3.9.13")
        # os.system(r"poetry env use C:\Users\pho\.pyenv\pyenv-win\versions\3.9.13\python.exe")
        os.system("python setup.py sdist bdist_wheel --dist-dir=./dist/")

    else:
        print(f'\t skipping building binary repos for {repo_path}')

    os.chdir('dist/')
    
    # Use glob to find the first generated .whl file in the dist/ directory
    found_whl_files = glob.glob('*.whl')
    found_whl_files = [a for a in found_whl_files if not a.endswith('current.whl')] # exclude the symlink from the search

    if len(found_whl_files) > 0:
        if len(found_whl_files) > 1:
            print(f'\t WARNING: multiple whl files found in dist/ directory: {found_whl_files}. Using the last one.')

        whl_file = found_whl_files[-1]
        # Symlink the whl file to a generic version:
        src_path = whl_file
        dst_path = 'current.whl'
        # Create the symbolic link
        try:
            print(f'\t symlinking {src_path} to {dst_path}')
            os.symlink(src_path, dst_path)
        except FileExistsError as e:
            print(f'\t WARNING: symlink {dst_path} already exists. Removing it.')
            # Remove the symlink
            os.unlink(dst_path)
            # Create the symlink
            os.symlink(src_path, dst_path)
        except Exception as e:
            raise e
    else:
        print(f'\t WARNING: No whl files found in {repo_path}')
        whl_file = None

    ## Restore initial directory by moving up a directory
    os.chdir('../')
    
    return whl_file

def setup_repo(repo_path, repo_url, is_binary_repo=False, is_release=False, enable_install_for_child_repos=False, enable_build_pyproject_toml=True, skip_lock_for_child_repos=False):
    """ Clones the repo if it doesn't exist, and updates it if it does."""
    print(f'=======> Processing Child Repo: `{repo_path}` ====]:')
    if not repo_path.exists():
        # clone the repo
        os.chdir(repo_path.parent) # change directory to the parent of the repo to prepare for cloning
        print(f'\t repo does not exist. Cloning {repo_path}...')
        os.system(f'git clone {repo_url}')
        os.chdir(repo_path)
    else:
        # update existing repo
        print(f'\t repo exists. Updating {repo_path}...')
        os.chdir(repo_path)
        GitHelpers.reset_local_changes(repo_path)
        # new files that are local only still hold things up
        os.system("git pull")

    if is_binary_repo:
        ## For binary repos:
        whl_file = _process_binary_repo(repo_path=repo_path, skip_building=args.skip_building_binary_repos)
        # if whl_file is not None:
        #     # update pyproject.toml file with the new version

    else:
        # For poetry repos
        _process_poetry_repo(repo_path, is_release=is_release, enable_build_pyproject_toml=enable_build_pyproject_toml, skip_lock=skip_lock_for_child_repos, enable_install=enable_install_for_child_repos)

    print(f'----------------------------------------------------------------------------------- done.\n')


# ==================================================================================================================== #
# Main Function                                                                                                        #
# ==================================================================================================================== #
def main():

    if args.release:
        print('Running in release mode')
    elif args.dev:
        print('Running in development mode')
    else:
        print('No mode selected')

    # if args.name:
    #     print(f'Hello, {args.name}!')
    # else:
    #     print('Hello, world!')

    force_overwrite_child_repos_from_remote = args.force
    skip_lock_for_child_repos = args.skip_lock
    is_release = (args.release == True)
    enable_install_for_child_repos = False


    ## Pull for children repos:
    dependent_repos = ["../NeuroPy", "../pyPhoCoreHelpers", "../pyPhoPlaceCellAnalysis"]
    dependent_repos_urls = ["https://github.com/CommanderPho/NeuroPy.git", "https://github.com/CommanderPho/pyPhoCoreHelpers.git", "https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git"]
    dependent_repos_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in dependent_repos]
    poetry_repo_tuples = list(zip(dependent_repos_paths, dependent_repos_urls, [False]*len(dependent_repos_paths), [False, True, True]))


    external_dependent_repos = ["../pyqode.core", "../pyqode.python", "../silx"]
    external_dependent_binary_repo_urls = ["https://github.com/CommanderPho/pyqode.core.git", "https://github.com/CommanderPho/pyqode.python.git", "https://github.com/CommanderPho/silx.git"]
    external_dependent_repo_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in external_dependent_repos]
    binary_repo_tuples = list(zip(external_dependent_repo_paths, external_dependent_binary_repo_urls, [True]*len(external_dependent_repo_paths), [False]*len(external_dependent_repo_paths)))

    os.system("git pull") # pull changes first for main repo
    print(f'{dependent_repos_paths = }')
    for i, (repo_path, repo_url, is_binary_repo, is_pyproject_toml_templated) in enumerate(poetry_repo_tuples + binary_repo_tuples):
        print(f'Updating {repo_path}...')
        setup_repo(repo_path, repo_url, is_binary_repo=is_binary_repo, is_release=is_release, enable_build_pyproject_toml=(is_pyproject_toml_templated and (not args.skip_building_templates)), skip_lock_for_child_repos=skip_lock_for_child_repos)

    os.chdir(root_dir) # change back to the root repo dir
    _process_poetry_repo(root_dir, is_release=is_release, enable_build_pyproject_toml=True, skip_lock=skip_lock_for_child_repos, enable_install=False)
    # os.system("poetry install --all-extras") # is this needed? I think it installs in that specific environment.
    print(f'done with all.')
    PoetryHelpers.install_ipython_kernel(kernel_name="spike3d-poetry") # run this to install the kernel for the poetry environment
    os.system("poetry run ipython kernel install --user --name=spike3d-poetry") # run this to install the kernel for the poetry environment

if __name__ == '__main__':
    main()

