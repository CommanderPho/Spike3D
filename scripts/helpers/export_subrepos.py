import os
from pathlib import Path
import glob # for finding .whl file after building binary repo
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


if __name__ == '__main__':
    main()