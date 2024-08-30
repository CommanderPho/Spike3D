import os
from pathlib import Path
import glob # for finding .whl file after building binary repo

from attrs import define, field
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray


# global_script_dir = Path(os.path.dirname(os.path.abspath(__file__))) # CHANGEME, used to be located `../../../scripts/` # ' /home/halechr/repos/Spike3D/scripts/helpers'
global_script_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent # CHANGEME, used to be located `../../../scripts/` # ' /home/halechr/repos/Spike3D/scripts/helpers'


# global_exports_folder = global_script_dir.joinpath('exports').resolve()
# print(f'script_dir: {global_script_dir}\nglobal_exports_folder: {global_exports_folder}')
# global_root_dir = global_script_dir.parent # Top-level parent root repo dir


@define(slots=False)
class SubrepoHelpers:
    """docstring for SubrepoHelpers.
    
    from poetry_helpers import PoetryHelpers
    from find_subrepos import SubrepoHelpers

    global_script_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent # CHANGEME, used to be located `../../../scripts/` # ' /home/halechr/repos/Spike3D/scripts/helpers'

    def main():
        print(f'Main')

        subrepos = SubrepoHelpers.init_from_script_dir(global_script_dir=global_script_dir)
        subrepos.global_exports_folder.mkdir(exist_ok=True)
        print(f'subrepos: {subrepos}')
        print(f'{subrepos.child_repos_paths = }')
        os.chdir(subrepos.global_root_dir)
        for i, (repo_name, repo_path) in enumerate(zip(subrepos.child_repos, subrepos.child_repos_paths)):
            print(f'Updating {repo_path}...')
            output_file_name = subrepos.global_exports_folder.joinpath(f'{repo_name}_requirements.txt')
            print(f'\t output_file_name: {output_file_name}') # f"../SCRIPTS/exports/requirements.txt"
            PoetryHelpers.export_poetry_repo(repo_path, output_file_path=output_file_name)
            # setup_repo(repo_path, repo_url, is_binary_repo=is_binary_repo, is_release=is_release, enable_build_pyproject_toml=(is_pyproject_toml_templated and (not args.skip_building_templates)), skip_lock_for_child_repos=skip_lock_for_child_repos)

        
    
    """
    global_script_dir: Path = field()
    global_Spike3D_root_dir: Path = field(init=False)
    global_exports_folder: Path = field(init=False)
    global_root_dir: Path = field(init=False)

    child_repos: List[str] = field(default=["NeuroPy", "pyPhoCoreHelpers", "pyPhoPlaceCellAnalysis", "Spike3D"])
    child_repos_urls: List[str] = field(default=["https://github.com/CommanderPho/NeuroPy.git", "https://github.com/CommanderPho/pyPhoCoreHelpers.git", "https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git", "https://github.com/CommanderPho/Spike3D.git"])
    child_repos_paths: List[Path] = field(init=False) # [global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in child_repos]

    external_child_repos: List[str] = field(default=["pyqode.core", "pyqode.python"])
    external_child_binary_repo_urls: List[str] = field(default=["https://github.com/CommanderPho/pyqode.core.git", "https://github.com/CommanderPho/pyqode.python.git"])
    external_child_repo_paths: List[Path] = field(init=False) # [global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in external_child_repos]


    def __attrs_post_init__(self):
        self.global_Spike3D_root_dir = self.global_script_dir.parent


        self.global_exports_folder = self.global_Spike3D_root_dir.parent.joinpath('exports').resolve()
        # self.global_exports_folder.mkdir(exist_ok=True)
        print(f'script_dir: {self.global_script_dir}\nglobal_exports_folder: {self.global_exports_folder}')
        self.global_root_dir = self.global_Spike3D_root_dir.parent # Top-level parent root repo dir

        # os.chdir(self.global_root_dir)
        print(f'global_Spike3D_root_dir: {self.global_Spike3D_root_dir}\nglobal_root_dir: {self.global_root_dir}')

        self.child_repos_paths = [self.global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in self.child_repos]
        self.external_child_repo_paths = [self.global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in self.external_child_repos]


    @classmethod
    def init_from_script_dir(cls, global_script_dir: Path) -> "SubrepoHelpers":
        return cls(global_script_dir=global_script_dir)


    @classmethod
    def reset_local_changes(cls, repo_path):
        """ Resets local changes to the repo."""
        os.chdir(repo_path)
        # os.system("git reset --hard HEAD")
        # os.system("git clean -f -d")
        os.system("git stash")
        os.system("git stash drop")
        




if __name__ == "__main__":
    subrepos = SubrepoHelpers.init_from_script_dir(global_script_dir=global_script_dir)
    subrepos.global_exports_folder.mkdir(exist_ok=True)
    print(f'subrepos: {subrepos}')