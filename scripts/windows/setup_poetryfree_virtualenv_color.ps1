param (
    [string]$colorFolderColorName
)

if (-not $colorFolderColorName) {
    $colorFolderColorName = Read-Host "Enter a color name (like 'yellow') without quotes:"
}

# Now you can use $colorFolderColorName in your script
Write-Output "Creating a new virtual-env with colorFolderColorName: $colorFolderColorName"

$envName = ".venv_$colorFolderColorName";

$envRelativeActivateScript = if ($env:OS -eq "Windows_NT") {
    "$envName\Scripts\activate"
} else {
    "$envName/bin/activate"
}
$envRelativePython = if ($env:OS -eq "Windows_NT") {
    "$envName\bin\python"
} else {
    "$envName/bin/python"
}

$fullEnvParentPath = if ($env:OS -eq "Windows_NT") {
    "K:\FastSwap\AppData\VSCode\$colorFolderColorName"
} else {
    "$HOME/Library/VSCode/$colorFolderColorName"
}

# mkdir -p $fullEnvParentPath
New-Item -ItemType Directory -Path $fullEnvParentPath -Force
cd "$fullEnvParentPath"

$fullEnvPath="$fullEnvParentPath\$envName"
$fullActivateScriptPath="$fullEnvParentPath\$envRelativeActivateScript"
$fullPythonPath="$fullEnvParentPath\$envRelativePython"

Write-Host "fullEnvParentPath: $fullEnvParentPath"
Write-Host "fullEnvPath: $fullEnvPath"
Write-Host "fullActivateScriptPath: $fullActivateScriptPath"
Write-Host "fullPythonPath: $fullPythonPath"

# Begin Body:
# cd "K:\FastSwap\AppData\VSCode\$colorFolderColorName"
cd "$fullEnvParentPath"
pyenv local 3.9.13
pyenv exec virtualenv $envName
deactivate
& $envRelativeActivateScript


$spike3d_repo_path="$HOME\repos\Spike3DWorkEnv\Spike3D"
Write-Host "Using spike3d_repo_path: $spike3d_repo_path..."

# symlink it
if (!(Test-Path $spike3d_repo_path -PathType Container)) {
    Write-Error "The directory $spike3d_repo_path does not exist. You will need to finish setup by symlinking and registering the kernel"
    # You can choose to create the directory here if desired
}
else {

    # Use the call operator (&) to execute the command with spaces
    & "$envName\Scripts\python" -m ensurepip --default-pip 
    & "$envName\Scripts\python" -m pip install --upgrade pip

    # Full-path:
    & "$envName\Scripts\python" -m pip install -r "C:\Users\pho\repos\Spike3DWorkEnv\Spike3D\requirements.txt"
    & "$envName\Scripts\python" -m pip install -e C:\Users\pho\repos\Spike3DWorkEnv\NeuroPy
    & "$envName\Scripts\python" -m pip install -e C:\Users\pho\repos\Spike3DWorkEnv\pyPhoCoreHelpers
    & "$envName\Scripts\python" -m pip install -e C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis

    # Link the new env as a symlink:
    $new_env_spike3D_folder_target="$spike3d_repo_path\$envName"
    echo "$new_env_spike3D_folder_target"
    New-Item -ItemType SymbolicLink -Path "$new_env_spike3D_folder_target" -Target "$fullEnvPath"
    cd "$new_env_spike3D_folder_target"
    echo "$new_env_spike3D_folder_target"

    ipython kernel install --user --name="spike3d-$colorFolderColorName"
    & "$fullActivateScriptPath"

    # Add to .gitignore:
    Add-Content -Path "$spike3d_repo_path\.gitignore" -Value "$envName"
    Write-Host "Completed successfully! done."
}



