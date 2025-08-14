import os
from pathlib import Path
import platform
import shutil


class GitHelpers:
    """docstring for GitHelpers."""
    
    @classmethod
    def reset_local_changes(cls, repo_path):
        """ Resets local changes to the repo."""
        os.chdir(repo_path)
        # os.system("git reset --hard HEAD")
        # os.system("git clean -f -d")
        os.system("git stash")
        os.system("git stash drop")
        
    @classmethod
    def create_pre_commit_script(cls, script_content, script_path):
        with open(script_path, 'w') as script_file:
            script_file.write(script_content)
        os.chmod(script_path, 0o755)

    @classmethod
    def install_hook(cls, repo_path: Path):
        # Define paths
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        
        assert repo_path.exists(), f"repo_path: '{repo_path}' does not exist!"
        dot_git_folder_dir = repo_path.joinpath('.git')
        assert (dot_git_folder_dir.exists() and dot_git_folder_dir.is_dir()), f"dot_git_folder_dir: '{dot_git_folder_dir}' does not exist for repo_path: '{repo_path}'! is it not a git directory?!?!"
        git_hooks_dir = dot_git_folder_dir.joinpath('hooks')
        git_hooks_dir.mkdir(mode=755, parents=False, exist_ok=True)
        # git_hooks_dir = repo_path.joinpath('.git', 'hooks')
        # git_hooks_dir = os.path.join('.git', 'hooks')
        # os.makedirs(git_hooks_dir, exist_ok=True)
        # notebook_path = "EXTERNAL/PhoDibaPaper2024Book/PhoDibaPaper2024.ipynb"		
        # Define common hook logic
        hook_logic = '''
        TARGET_NOTEBOOK="EXTERNAL/PhoDibaPaper2024Book/PhoDibaPaper2024.ipynb"
        if git diff --cached --name-only | grep -q "$TARGET_NOTEBOOK"; then
            nbstripout "$TARGET_NOTEBOOK"
            git add "$TARGET_NOTEBOOK"
        fi
        '''
        if platform.system() == "Windows":
            # PowerShell script content
            powershell_content = f"""
            $TARGET_NOTEBOOK = "EXTERNAL/PhoDibaPaper2024Book/PhoDibaPaper2024.ipynb"
            if ((git diff --cached --name-only) -contains $TARGET_NOTEBOOK) {{
                nbstripout $TARGET_NOTEBOOK
                git add $TARGET_NOTEBOOK
            }}
            """
            script_path = git_hooks_dir.joinpath('pre-commit.ps1')
            cls.create_pre_commit_script(powershell_content, script_path)
            print(f"PowerShell pre-commit hook installed at {script_path}")
        else:
            # Bash script content
            bash_content = f"#!/bin/sh\n{hook_logic}"
            script_path = git_hooks_dir.joinpath('pre-commit')
            cls.create_pre_commit_script(bash_content, script_path)
            print(f"Bash pre-commit hook installed at {script_path}")


if __name__ == "__main__":
    repo_path = "C:/Users/pho/repos/Spike3DWorkEnv/Spike3D"
    GitHelpers.install_hook(repo_path=repo_path)
    
