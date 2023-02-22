Since I ran into so many issues adding pyqode as dependency using Poetry, I just used each repo's natural build/setup process to build a binary distribution (`.whl`) for the platform and then added that as a poetry requirement.

## Built binary distributions for both `pyqode_core` and `haesleinhuepf_pyqode.python`:
```bash

C:/Users/pho/AppData/Local/pypoetry/Cache/virtualenvs/pyphoplacecellanalysis-FtPLvXd1-py3.9/Scripts/activate.bat

cd "C:\Users\pho\repos\pyqode.core"
python setup.py sdist bdist_wheel

cd "C:\Users\pho\repos\ExternalTesting\pyqode.python"
python setup.py sdist bdist_wheel

```

## Added the binary .whl files to poetry as a dependency:

```bash
poetry add C:\Users\pho\repos\pyqode.core\dist\pyqode_core-3.0.0-py2.py3-none-any.whl C:\Users\pho\repos\ExternalTesting\pyqode.python\dist\haesleinhuepf_pyqode.python-2.15.2-py2.py3-none-any.whl

poetry lock
poetry install
```
