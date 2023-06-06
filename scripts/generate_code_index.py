import subprocess
import pathlib


from scripts.helpers.source_code_helpers import find_py_files


def build_code_index(project_path, exclude_dirs=[]):
    """ 2023-05-11 - Tries to build a CodeQuery code index. Not quite working.
    
    Original Bash Commands:
    

        cd "C:/Users/pho/repos/Spike3DWorkEnv/NeuroPy/neuropy"
        dir /b/a/s *.py    > cscope.files
        pycscope -i cscope.files
        ctags --fields=+i -n -L cscope.files
        cqmakedb -s .\myproject.db -c cscope.out -t tags -p


        cd "C:/Users/pho/repos/Spike3DWorkEnv/pyPhoCoreHelpers/src"
        dir /b/a/s *.py    > cscope.files
        pycscope -i cscope.files
        ctags --fields=+i -n -L cscope.files
        cqmakedb -s .\myproject.db -c cscope.out -t tags -p


        cd "C:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src"
        dir /b/a/s *.py | find /i /v "External\\" | findstr /i /v "\\External\\"   > cscope.files 

        # NOTE: Works when excluding all External/* files
        pycscope -i cscope.files
        ctags --fields=+i -n -L cscope.files
        cqmakedb -s .\myproject.db -c cscope.out -t tags -p

        
        # This does not work: the idea of adding them all to a shared database
        cqmakedb -s ..\..\full_project.db -c cscope.out -t tags -p

        
        
        
    
    Example:
        project_path = "C:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src"
        exclude_dirs = ["pyphoplacecellanalysis/External"]

    """
    project_path = pathlib.Path(project_path)
    cscope_files = project_path / "cscope.files"
    cscope_out = project_path / "cscope.out"
    tags = project_path / "tags"
    db_path = project_path / "myproject.db"

    # Find all .py files in the project directory and its subdirectories
    included_py_files = find_py_files(project_path, exclude_dirs=exclude_dirs)

    # Write the file paths to cscope.files
    with open(project_path / "cscope.files", "w") as f:
        for file_path in included_py_files:
            f.write(str(file_path) + "\n")

    # Generate the cscope index
    subprocess.run(["pycscope", "-i", str(cscope_files)], cwd=str(project_path), check=True)
    
    # Generate the ctags index
    subprocess.run(["ctags", "--fields=+i", "-n", "-L", str(cscope_files)], cwd=str(project_path), check=True)
    
    # Generate the CodeQL database
    subprocess.run(["cqmakedb", "-s", str(db_path), "-c", str(cscope_out), "-t", str(tags), "-p"], cwd=str(project_path), check=True)


if __name__ == "__main__":
    build_code_index("C:/Users/pho/repos/Spike3DWorkEnv/NeuroPy/neuropy")
    build_code_index("C:/Users/pho/repos/Spike3DWorkEnv/pyPhoCoreHelpers/src")
    build_code_index("C:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src", exclude_dirs = ["pyphoplacecellanalysis/External"])





# "C:\Program Files\Microsoft VS Code\Code.exe" --goto "%f:%n"