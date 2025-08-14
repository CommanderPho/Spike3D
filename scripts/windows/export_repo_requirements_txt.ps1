# Exports a requirements.txt suitable for use by pip/virtualenv from each repo's poetry project:

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

$directory_poetry_export_commands = @(
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt"),
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt"),
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt"),
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt")
)

function Export-RequirementsFromPoetry {
    Write-Host "Exporting requirements.txt from poetry repos..."
    for ($i = 0; $i -lt $directories.Length; $i++) {
        $directory = $directories[$i]
        $commands = $directory_poetry_export_commands[$i]
        Set-Location $directory
        Write-Host "\t for $directory"
        foreach ($command in $commands) {
            Invoke-Expression $command
        }
        Write-Host "done."
    }
}


Export-RequirementsFromPoetry
