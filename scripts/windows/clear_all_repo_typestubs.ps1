# Exports typestubs from each repo:

$spike3DDirectory = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) # PSScriptRoot is the script path
Write-Verbose "spike3DDirectory: $spike3DDirectory" # C:\Users\pho\repos\Spike3DWorkEnv\Spike3D
$repoParentDirectory = Split-Path -Parent $spike3DDirectory
Write-Verbose "repoParentDirectory: $repoParentDirectory" #  C:\Users\pho\repos\Spike3DWorkEnv

$directories = @(
    "$repoParentDirectory\NeuroPy",
    "$repoParentDirectory\pyPhoCoreHelpers",
    "$repoParentDirectory\pyPhoPlaceCellAnalysis",
    "$spike3DDirectory"
)

# "Remove-Item -Recurse -Force .\typings\", 

function Clear-PyrightTypestubs {
    Write-Host "Clearing (Deleting) all existing typestubs..."
    for ($i = 0; $i -lt $directories.Length; $i++) {
        $directory = $directories[$i]
        Set-Location $directory
        Write-Host "    for $directory"
        Remove-Item -Recurse -Force  -ErrorAction SilentlyContinue "$directory\typings"
        Write-Host "done."
    }
}


Clear-PyrightTypestubs