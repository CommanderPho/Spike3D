2023-06-22

This file serves to document some of the weirdness involving Poetry virtual environments.


# Normal .venv directory:

(`virtualenvs.in-project` == False)
```bash
(spike3d-py3.9) [halechr@LNX00052 Spike3D]$ poetry env list
spike3d-u9wuHq2d-py3.11
spike3d-u9wuHq2d-py3.9 (Activated)
(spike3d-py3.9) [halechr@LNX00052 Spike3D]$ poetry env info

Virtualenv
Python:         3.9.13
Implementation: CPython
Path:           /home/halechr/.cache/pypoetry/virtualenvs/spike3d-u9wuHq2d-py3.9
Executable:     /home/halechr/.cache/pypoetry/virtualenvs/spike3d-u9wuHq2d-py3.9/bin/python
Valid:          True
```

📦spike3d-u9wuHq2d-py3.9
 ┣ 📂bin
 ┃ ┣ 📜activate
 ┃ ┣ 📜activate.csh
 ┃ ┣ 📜activate.fish
 ┃ ┣ 📜activate.nu
 ┃ ┣ 📜activate.ps1
 ┃ ┣ 📜activate_this.py
 ┃ ┣ 📜ansi2html
...
 ┃ ┣ 📜jupyter-kernel
 ┃ ┣ 📜jupyter-kernelspec
 ┃ ┣ 📜jupyter-lab
...
 ┃ ┣ 📜python
 ┃ ┣ 📜python3
 ┃ ┣ 📜python3.9
 ...
 ┣ 📂etc
 ┃ ┗ 📂jupyter
 ┣ 📂include
 ┣ 📂lib
 ┃ ┗ 📂python3.9
 ┣ 📂lib64
 ┣ 📂share
 ┃ ┣ 📂applications
 ┃ ┣ 📂doc
 ┃ ┣ 📂icons
 ┃ ┣ 📂jupyter
 ┃ ┗ 📂man
 ┣ 📂silx
 ┃ ┗ 📂third_party
 ┣ 📂src
 ┃ ┣ 📂ansi2html
 ┃ ┣ 📂mpl-multitab
 ┃ ┣ 📂pybursts
 ┃ ┗ 📂vedo
 ┣ 📜.gitignore
 ┣ 📜CMakeLists.txt
 ┣ 📜LICENSE.txt
 ┣ 📜pyproject.toml
 ┗ 📜pyvenv.cfg


source /home/halechr/.cache/pypoetry/virtualenvs/spike3d-u9wuHq2d-py3.9/bin/activate



# Project-folder specific .venv directory:
(`virtualenvs.in-project` == True)
📦.venv
 ┣ 📂bin
 ┃ ┣ 📜activate
 ┃ ┣ 📜activate.csh
 ┃ ┣ 📜activate.fish
 ┃ ┣ 📜activate.nu
 ┃ ┣ 📜activate.ps1
 ┃ ┣ 📜activate_this.py
 ┃ ┣ 📜ansi2html
 ...
 ┣ 📂etc
 ┃ ┗ 📂jupyter
 ┣ 📂include
 ┃ ┣ 📂blosc2
 ┃ ┣ 📂site
 ┃ ┗ 📜blosc2.h
 ┣ 📂lib
 ┃ ┗ 📂python3.9
 ┣ 📂lib64
 ┣ 📂share
 ┃ ┣ 📂applications
 ┃ ┣ 📂doc
 ┃...
 ┣ 📜.gitignore
 ┣ 📜CMakeLists.txt
 ┣ 📜LICENSE.txt
 ┣ 📜pyproject.toml
 ┗ 📜pyvenv.cfg


Spawning shell within /home/halechr/repo/Spike3D/.venv
. /home/halechr/repo/Spike3D/.venv/bin/activate
