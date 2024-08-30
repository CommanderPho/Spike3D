import os
import sys
import subprocess
import shutil
from enum import Enum
from pathlib import Path
import glob

# from pyphocorehelpers.Filesystem.path_helpers import copy_recursive

# from helpers.path_helpers import copy_recursive
# from helpers.source_code_helpers import replace_text_in_file # for finding .whl file after building binary repo

from path_helpers import copy_recursive
from source_code_helpers import replace_text_in_file # for finding .whl file after building binary repo



# ==================================================================================================================== #
# Project versioning:                                                                                                  #
# ==================================================================================================================== #

# pyproject_files = {'release':'pyproject_release.toml', 'dev':'pyproject_dev.toml'}


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


class PoetryHelpers:
    """ 
    from scripts.helpers.poetry_helpers import PoetryHelpers, VersionType

    """
    def build_pyproject_toml_file(repo_path, is_release=False, pyproject_final_file_name = 'pyproject.toml', debug_print=False):
        """ Builds the complete final pyproject.toml file from the pyproject_template.toml_template for the current version (release or dev)

        from Spike3D.scripts.setup_dependent_repos import build_pyproject_toml_file
        build_pyproject_toml_file("C:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis")

        """
        os.chdir(repo_path)
        curr_version = VersionType.init_from_is_release(is_release)
        print(f'\t Templating: Building pyproject.toml for {curr_version.name} version in {repo_path}...')
        remote_dependencies_regex = r"^(\s*\[tool\.poetry\.group\.remote\.dependencies\]\n(?:.+\n)*)\n\[?"
        # Load the insert text
        with open(curr_version.pyproject_template_file, 'r') as f:
            insert_text_str = f.read()
            
        if not insert_text_str.startswith('\n'):
            # Add a leading newline if the loaded text doesn't already have one
            insert_text_str = '\n' + insert_text_str
        if not insert_text_str.endswith('\n\n'):
            # Add a trailing newline if the loaded text doesn't already have one
            insert_text_str = insert_text_str + '\n'
        
        if debug_print:
            print(insert_text_str)
        
        replace_text_in_file(pyproject_final_file_name, remote_dependencies_regex, insert_text_str, debug_print=debug_print)
        return pyproject_final_file_name




    @classmethod
    def get_current_poetry_env(cls, working_dir=None) -> str:
        if working_dir is not None:
            curr_env_path = subprocess.check_output([f'poetry env list --full-path --directory={working_dir}'], shell=True)
        else:
            curr_env_path = subprocess.check_output(['poetry env list --full-path'], shell=True) # use the current working directory
        return curr_env_path

    @staticmethod
    def install_ipython_kernel(kernel_name:str="spike3d-poetry"):
        """ run this to install the kernel for the poetry environment """
        os.system(f"poetry run ipython kernel install --user --name={kernel_name}") # run this to install the kernel for the poetry environment


    @classmethod
    def backup_poetry_env(cls):
        """ backs up the current poetry environment to a zip file and returns the path of the file. """
        raise NotImplementedError

    @classmethod
    def export_poetry_repo(cls, repo_path, output_file_path="requirements.txt"):
        """ Exports the child repo.
        
        from Spike3D.scripts.helpers.export_subrepos import export_poetry_repo
        
        PoetryHelpers.export_poetry_repo(
        """
        if not isinstance(repo_path, Path):
            repo_path = Path(repo_path).resolve()
        print(f'=======> Processing Child Repo: `{repo_path}` ====]:')
        assert repo_path.exists()
        os.chdir(repo_path)
        export_command = f'poetry export --without-hashes --format=requirements.txt > "{output_file_path}"'
        print(f'\texport_command: {export_command}')
        os.system(export_command) # run this to install the kernel for the poetry environment
        print(f'----------------------------------------------------------------------------------- done.\n')


    @classmethod
    def duplicate_poetry_env(cls):
        # Get the path to the current Poetry environment
        env_path = subprocess.check_output(['poetry', 'env', 'info', '--path']).decode().strip()

        # Create a new environment with the same packages as the current environment
        new_env_path = os.path.join(os.path.dirname(env_path), 'new_env')
        subprocess.run(['python', '-m', 'venv', new_env_path])
        subprocess.run([os.path.join(new_env_path, 'bin', 'pip'), 'install', '-r', os.path.join(env_path, 'requirements.txt')])

        return new_env_path



r""" 
cache-dir = "W:\\FastSwap\\pypoetry_CACHE"
experimental.system-git-client = false
installer.max-workers = null
installer.modern-installation = true
installer.no-binary = null
installer.parallel = true
virtualenvs.create = true
virtualenvs.in-project = true
virtualenvs.options.always-copy = true
virtualenvs.options.no-pip = false
virtualenvs.options.no-setuptools = false
virtualenvs.options.system-site-packages = false
virtualenvs.path = "{cache-dir}\\virtualenvs"  # W:\FastSwap\pypoetry_CACHE\virtualenvs
virtualenvs.prefer-active-python = true
virtualenvs.prompt = "{project_name}-py{python_version}"
"""

r""" TODO: turn into poetry commands:

(spike3d-py3.9) C:\Users\pho\repos\Spike3DWorkEnv\Spike3D>poetry env info --executable
C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\Scripts\python.exe

(spike3d-py3.9) C:\Users\pho\repos\Spike3DWorkEnv\Spike3D>poetry env info --path
C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv

(spike3d-py3.9) C:\Users\pho\repos\Spike3DWorkEnv\Spike3D>poetry env list --full-path
C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv (Activated)
W:\FastSwap\pypoetry_CACHE\virtualenvs\spike3d-UP7QTzFM-py3.9


"""

# global_script_dir = Path(os.path.dirname(os.path.abspath(__file__))) #
# global_exports_folder = global_script_dir.joinpath('exports').resolve()
# print(f'script_dir: {global_script_dir}\nglobal_exports_folder: {global_exports_folder}')
# global_root_dir = global_script_dir.parent # Top-level parent root repo dir
# os.chdir(global_root_dir)

# child_repos = ["NeuroPy", "pyPhoCoreHelpers", "pyPhoPlaceCellAnalysis", "Spike3D"]
# child_repos_urls = ["https://github.com/CommanderPho/NeuroPy.git", "https://github.com/CommanderPho/pyPhoCoreHelpers.git", "https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git", "https://github.com/CommanderPho/Spike3D.git"]
# child_repos_paths = [global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in child_repos]


# external_child_repos = ["pyqode.core", "pyqode.python"]
# external_child_binary_repo_urls = ["https://github.com/CommanderPho/pyqode.core.git", "https://github.com/CommanderPho/pyqode.python.git"]
# external_child_repo_paths = [global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in external_child_repos]


# def export_poetry_repo(repo_path, output_file_path="requirements.txt"):
# 	""" Exports the child repo. """
# 	print(f'=======> Processing Child Repo: `{repo_path}` ====]:')
# 	assert repo_path.exists()
# 	os.chdir(repo_path)
# 	export_command = f'poetry export --without-hashes --format=requirements.txt > "{output_file_path}"'
# 	print(f'\texport_command: {export_command}')
# 	os.system(export_command) # run this to install the kernel for the poetry environment
# 	print(f'----------------------------------------------------------------------------------- done.\n')



# def main():
#     print(f'Main')
#     print(f'{child_repos_paths = }')
#     for i, (repo_name, repo_path) in enumerate(zip(child_repos, child_repos_paths)):
#         print(f'Updating {repo_path}...')
#         output_file_name = global_exports_folder.joinpath(f'{repo_name}_requirements.txt')
#         print(f'\t output_file_name: {output_file_name}') # f"../SCRIPTS/exports/requirements.txt"
#         export_poetry_repo(repo_path, output_file_path=output_file_name)
#         # setup_repo(repo_path, repo_url, is_binary_repo=is_binary_repo, is_release=is_release, enable_build_pyproject_toml=(is_pyproject_toml_templated and (not args.skip_building_templates)), skip_lock_for_child_repos=skip_lock_for_child_repos)



# if __name__ == '__main__':
#     main()