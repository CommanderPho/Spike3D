import os
from pathlib import Path
import glob # for finding .whl file after building binary repo


global_script_dir = Path(os.path.dirname(os.path.abspath(__file__))) # CHANGEME, used to be located `../../../scripts/`
global_exports_folder = global_script_dir.joinpath('exports').resolve()
print(f'script_dir: {global_script_dir}\nglobal_exports_folder: {global_exports_folder}')
global_root_dir = global_script_dir.parent # Top-level parent root repo dir
os.chdir(global_root_dir)

child_repos = ["NeuroPy", "pyPhoCoreHelpers", "pyPhoPlaceCellAnalysis", "Spike3D"]
child_repos_urls = ["https://github.com/CommanderPho/NeuroPy.git", "https://github.com/CommanderPho/pyPhoCoreHelpers.git", "https://github.com/CommanderPho/pyPhoPlaceCellAnalysis.git", "https://github.com/CommanderPho/Spike3D.git"]
child_repos_paths = [global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in child_repos]


external_child_repos = ["pyqode.core", "pyqode.python"]
external_child_binary_repo_urls = ["https://github.com/CommanderPho/pyqode.core.git", "https://github.com/CommanderPho/pyqode.python.git"]
external_child_repo_paths = [global_root_dir.joinpath(a_rel_path).resolve() for a_rel_path in external_child_repos]


def export_poetry_repo(repo_path, output_file_path="requirements.txt"):
	""" Exports the child repo.
     
    from Spike3D.scripts.helpers.export_subrepos import export_poetry_repo
     
    """
	print(f'=======> Processing Child Repo: `{repo_path}` ====]:')
	assert repo_path.exists()
	os.chdir(repo_path)
	export_command = f'poetry export --without-hashes --format=requirements.txt > "{output_file_path}"'
	print(f'\texport_command: {export_command}')
	os.system(export_command) # run this to install the kernel for the poetry environment
	print(f'----------------------------------------------------------------------------------- done.\n')



def main():
    print(f'Main')
    print(f'{child_repos_paths = }')
    for i, (repo_name, repo_path) in enumerate(zip(child_repos, child_repos_paths)):
        print(f'Updating {repo_path}...')
        output_file_name = global_exports_folder.joinpath(f'{repo_name}_requirements.txt')
        print(f'\t output_file_name: {output_file_name}') # f"../SCRIPTS/exports/requirements.txt"
        export_poetry_repo(repo_path, output_file_path=output_file_name)
        # setup_repo(repo_path, repo_url, is_binary_repo=is_binary_repo, is_release=is_release, enable_build_pyproject_toml=(is_pyproject_toml_templated and (not args.skip_building_templates)), skip_lock_for_child_repos=skip_lock_for_child_repos)


	




if __name__ == '__main__':
    main()