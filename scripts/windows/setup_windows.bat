@REM Install Chocolatey
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"

@REM Use Chocolatey to install required system utilities
choco install vscode git curl dvc pyenv-win -y # installs vscode, git, dvc, and pyenv-win

@REM Install Python 3.9.13
refreshEnv
pyenv install 3.9.13
pyenv shell 3.9.13
pyenv local 3.9.13 @REM this sets the new install to be the local Python version (in the directory where this command is ran)

@REM Install Poetry
curl -sSL https://install.python-poetry.org | python3 - @REM get poetry and install it

@REM add Poetry to path for current shell only
set PATH=%PATH%;%APPDATA%\Python\Scripts
poetry env use C:\Users\pho\.pyenv\pyenv-win\versions\3.9.13\python.exe
poetry shell

