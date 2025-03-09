
  Id     Duration CommandLine
  --     -------- -----------
   1        0.003 cd .\repos\Spike3DWorkEnv\
   2        0.003 cd Spike3D
   3        0.556 poetry env list --full-path
   4        0.532 poetry env activate
   5        0.039 & "C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\.venv\Scripts\Activate.ps1"
   6       37.169 .\scripts\windows\export_requirements_and_create_typestubs.ps1
   7     4:04.814 C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\scripts\windows\setup_poetryfree_virtualenv_color.ps1
   8        0.407 poetry install --all-extras
   9        0.003 cd ../
  10     1:59.141 poetry install --all-extras
  11     1:22.371 poetry install --no-root
  12        9.798 poetry install --all-extras
  13       32.229 poetry add PyQtWebEngine-Qt5
  14       26.286 poetry add ..\PyQtWebEngine-5.15.7.tar.gz
  15        5.502 pip wheel --no-cache-dir --use-pep517 "pyqtwebengine @ file:///C:/Users/pho/repos/Spike3DWorkEnv/PyQtWebEngine-5.15.7.tar.gz"
  16        3.201 python.exe -m pip install --upgrade pip
  17        4.082 python.exe -m pip install --upgrade PyQtWebEngine
  18        0.012 history

