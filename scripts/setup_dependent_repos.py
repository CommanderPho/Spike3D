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

is_release = False
enable_install_for_child_repos = False

script_dir = Path(os.path.dirname(os.path.abspath(__file__))) # /home/halechr/repos/Spike3D/scripts
print(f'script_dir: {script_dir}')
root_dir = script_dir.parent # Spike3D root repo dir
os.chdir(root_dir)

def insert_text(source_file, insert_text_str:str, output_file, insertion_string:str='<INSERT_HERE>'):
    """Inserts the text from insert_text_str into the source_file at the insertion_string, and saves the result to output_file.

    Args:
        source_file (_type_): _description_
        insert_text_str (str): _description_
        output_file (_type_): _description_
        insertion_string (str, optional): _description_. Defaults to '<INSERT_HERE>'.
    """
    # Load the source text
    with open(source_file, 'r') as f:
        source_text = f.read()

    # Find the insertion point in the source text
    insert_index = source_text.find(insertion_string)

    # Insert the text
    updated_text = source_text[:insert_index] + insert_text_str + source_text[insert_index:]

    # Save the updated text to the output file
    with open(output_file, 'w') as f:
        f.write(updated_text)

def insert_text_from_file(source_file, insert_file, output_file, insertion_string:str='<INSERT_HERE>'):
    """ Wraps insert_text, but loads the insert_text from a file instead of a string. """
    # Load the insert text
    with open(insert_file, 'r') as f:
        insert_text = f.read()
    insert_text(source_file, insert_text, output_file, insertion_string)


# ==================================================================================================================== #
# Project versioning:                                                                                                  #
# ==================================================================================================================== #

# pyproject_files = {'release':'pyproject_release.toml', 'dev':'pyproject_dev.toml'}


from enum import Enum

class VersionType(Enum):
    """Docstring for VersionType."""
    RELEASE = "release"
    dev = "dev"
    
    @property
    def pyproject_template_file(self):
        """The pyproject_exclusive_text property."""
        filename = {'release': "templating/pyproject_template_release.toml_fragment",
        'dev': "templating/pyproject_template_dev.toml_fragment"}
        if self.name == VersionType.RELEASE.name:
            return filename['release']
        elif self.name == VersionType.dev.name:
            return filename['dev']
        else:
            raise ValueError(f"VersionType {self.name} not recognized.")

    @property
    def pyproject_exclusive_text(self):
        """The pyproject_exclusive_text property."""
        _pyproject_exclusive_text_dict = {'release': """[tool.poetry.group.local.dependencies]
        neuropy = {path = "../NeuroPy", develop=true} # , extras = ["acceleration"]
        pyphocorehelpers = {path = "../pyPhoCoreHelpers", develop=true}
        pyphoplacecellanalysis = {path = "../pyPhoPlaceCellAnalysis", develop=true}""",
        'dev': """[tool.poetry.group.local.dependencies]
        neuropy = {path = "../NeuroPy", develop=true} # , extras = ["acceleration"]
        pyphocorehelpers = {path = "../pyPhoCoreHelpers", develop=true}
        pyphoplacecellanalysis = {path = "../pyPhoPlaceCellAnalysis", develop=true}"""}
        if self.name == VersionType.RELEASE.name:
            return _pyproject_exclusive_text_dict['release']
        elif self.name == VersionType.dev.name:
            return _pyproject_exclusive_text_dict['dev']
        else:
            raise ValueError(f"VersionType {self.name} not recognized.")
    
    @classmethod
    def init_from_is_release(cls, is_release: bool):
        if is_release:
            return cls.RELEASE
        else:
            return cls.dev


def build_pyproject_toml_file(repo_path, is_release=False, pyproject_template_file_name = 'templating/pyproject_template.toml_template', pyproject_final_file_name = 'pyproject.toml'):
    """ Builds the complete final pyproject.toml file from the pyproject_template.toml_template for the current version (release or dev) """
    os.chdir(repo_path)
    curr_version = VersionType.init_from_is_release(is_release)

    print(f'building pyproject.toml for {curr_version.name} version.')
    # insert_text(pyproject_template_file_name, curr_version.pyproject_exclusive_text, pyproject_final_file_name, insertion_string='<INSERT_HERE>')
    insert_text_from_file(pyproject_template_file_name, curr_version.pyproject_template_file, pyproject_final_file_name, insertion_string='<INSERT_HERE>')
    # if is_release:
    #     os.system(f"cp {pyproject_files['release']} {pyproject_final_file_name}")
    # else:
    #     os.system(f"cp {pyproject_files['dev']} {pyproject_final_file_name}")

# # alternative method of commenting out
# "RUN sed -i -n '/tool.poetry.dev-dependencies/q;p' pyproject.toml"



def _reset_local_changes(repo_path):
    """ Resets local changes to the repo."""
    os.chdir(repo_path)
    # os.system("git reset --hard HEAD")
    # os.system("git clean -f -d")
    os.system("git stash")
    os.system("git stash drop")


def process_poetry_repo(repo_path, is_release=False, enable_build_pyproject_toml=True, skip_lock=False, enable_install=False):
    ## Build final pyproj.toml file
    if enable_build_pyproject_toml:
        build_pyproject_toml_file(repo_path, is_release=is_release)
    else:
        print(f'skipping build pyproject.toml for {repo_path}')
    if not skip_lock:
        os.system("poetry lock")
    else:
        print(f'skipping lock for {repo_path}')
    if enable_install:
        os.system("poetry install") # is this needed? I think it installs in that specific environment.


def setup_repo(repo_path, repo_url, is_binary_repo=False, is_release=False, enable_install_for_child_repos=False, enable_build_pyproject_toml=True, skip_lock_for_child_repos=False):
    """ Clones the repo if it doesn't exist, and updates it if it does."""
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
        _reset_local_changes(repo_path)
        # new files that are local only still hold things up
        os.system("git pull")

    if is_binary_repo:
        ## For binary repos:
        # os.system("pyenv local 3.9.13")
        # os.system(r"poetry env use C:\Users\pho\.pyenv\pyenv-win\versions\3.9.13\python.exe")
        os.system("python setup.py sdist bdist_wheel")
    else:
        # For poetry repos
        process_poetry_repo(repo_path, is_release=is_release, enable_build_pyproject_toml=enable_build_pyproject_toml, skip_lock=skip_lock_for_child_repos, enable_install=enable_install_for_child_repos)

def main():
    
    ## Pull for children repos:
    dependent_repos = ["../NeuroPy", "../pyPhoCoreHelpers", "../pyPhoPlaceCellAnalysis"]
    dependent_repos_urls = ["https://github.com/CommanderPho/NeuroPy.git", "https://github.com/CommanderPho/pyPhoCoreHelpers.git", "https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git"]
    dependent_repos_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in dependent_repos]
    poetry_repo_tuples = list(zip(dependent_repos_paths, dependent_repos_urls, [False]*len(dependent_repos_paths), [False, True, True]))


    external_dependent_repos = ["../pyqode.core", "../pyqode.python"]
    external_dependent_binary_repo_urls = ["https://github.com/CommanderPho/pyqode.core.git", "https://github.com/CommanderPho/pyqode.python.git"]
    external_dependent_repo_paths = [root_dir.joinpath(a_rel_path).resolve() for a_rel_path in external_dependent_repos]
    binary_repo_tuples = list(zip(external_dependent_repo_paths, external_dependent_binary_repo_urls, [True]*len(external_dependent_repo_paths), [False]*len(external_dependent_repo_paths)))

    os.system("git pull") # pull changes first for main repo
    print(f'{dependent_repos_paths = }')
    for i, (repo_path, repo_url, is_binary_repo, is_pyproject_toml_templated) in enumerate(poetry_repo_tuples + binary_repo_tuples):
        print(f'Updating {repo_path}...')
        setup_repo(repo_path, repo_url, is_binary_repo=is_binary_repo, is_release=is_release, enable_build_pyproject_toml=is_pyproject_toml_templated, skip_lock_for_child_repos=True)

    os.chdir(root_dir) # change back to the root repo dir
    os.system("poetry lock")
    process_poetry_repo(root_dir, is_release=is_release, enable_build_pyproject_toml=True, skip_lock=True, enable_install=False)
    os.system("poetry install --all-extras") # is this needed? I think it installs in that specific environment.
    print(f'done with all.')

    os.system("poetry run ipython kernel install --user --name=spike3d-poetry") # run this to install the kernel for the poetry environment

if __name__ == '__main__':
    main()

