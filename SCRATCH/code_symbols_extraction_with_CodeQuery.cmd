ctags -R --languages=python .

C:/Users/pho/repos/Spike3DWorkEnv/NeuroPy/neuropy
C:/Users/pho/repos/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers
C:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis
C:/Users/pho/repos/Spike3DWorkEnv/Spike3D

ctags -R --languages=python C:/Users/pho/repos/Spike3DWorkEnv/NeuroPy/neuropy C:/Users/pho/repos/Spike3DWorkEnv/pyPhoCoreHelpers/src/pyphocorehelpers C:/Users/pho/repos/Spike3DWorkEnv/pyPhoPlaceCellAnalysis/src/pyphoplacecellanalysis C:/Users/pho/repos/Spike3DWorkEnv/Spike3D




ctags -R --languages=python C:\Users\pho\repos\Spike3DWorkEnv\NeuroPy\neuropy C:\Users\pho\repos\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis C:\Users\pho\repos\Spike3DWorkEnv\Spike3D -f SCRIPTS\.tags


pycscope -R C:\Users\pho\repos\Spike3DWorkEnv\NeuroPy\neuropy C:\Users\pho\repos\Spike3DWorkEnv\pyPhoCoreHelpers\src\pyphocorehelpers C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis C:\Users\pho\repos\Spike3DWorkEnv\Spike3D -f SCRIPTS\pycscope.out


## On Linux:
find . -iname "*.py" > ./cscope.files

## On Windows:
dir /b/a/s *.py    > cscope.files


Search 

`In print(f'') strings`
In strings (between "")
In comments



python build symbol tree GUI utility

cd "C:\Users\pho\repos\Spike3DWorkEnv\NeuroPy\neuropy"
dir /b/a/s *.py    > cscope.files
pycscope -i cscope.files
ctags --fields=+i -n -L cscope.files
cqmakedb -s .\myproject.db -c cscope.out -t tags -p


cd "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoCoreHelpers\src"
dir /b/a/s *.py    > cscope.files
pycscope -i cscope.files
ctags --fields=+i -n -L cscope.files
cqmakedb -s .\myproject.db -c cscope.out -t tags -p


cd "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis\src"
dir /b/a/s *.py | find /i /v "External\\" | findstr /i /v "\\External\\"   > cscope.files 


# NOTE: Works when excluding all External/* files
pycscope -i cscope.files
ctags --fields=+i -n -L cscope.files
cqmakedb -s .\myproject.db -c cscope.out -t tags -p


# This does not work: the idea of adding them all to a shared database
cqmakedb -s ..\..\full_project.db -c cscope.out -t tags -p


```python

import subprocess
import pathlib

def build_code_index(project_path):
    project_path = pathlib.Path(project_path)
    cscope_files = project_path / "cscope.files"
    cscope_out = project_path / "cscope.out"
    tags = project_path / "tags"
    db_path = project_path / "myproject.db"
    
    # Generate the cscope index
    subprocess.run(["pycscope", "-i", str(cscope_files)], cwd=str(project_path), check=True)
    
    # Generate the ctags index
    subprocess.run(["ctags", "--fields=+i", "-n", "-L", str(cscope_files)], cwd=str(project_path), check=True)
    
    # Generate the CodeQL database
    subprocess.run(["cqmakedb", "-s", str(db_path), "-c", str(cscope_out), "-t", str(tags), "-p"], cwd=str(project_path), check=True)




build_code_index("C:/Users/pho/repos/Spike3DWorkEnv/pyPhoCoreHelpers/src")

```