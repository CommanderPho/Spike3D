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

$directory_pyright_generate_typestubs_commands = @(
    @("pyright --createstub neuropy"),
    @("pyright --createstub pyphocorehelpers"),
    @("pyright --createstub pyphoplacecellanalysis"),
    @("pyright --createstub neuropy", "pyright --createstub pyphocorehelpers", "pyright --createstub pyphoplacecellanalysis")
)

function Export-PyrightTypestubs {
    Write-Host "Exporting typestubs using PyRight..."
    for ($i = 0; $i -lt $directories.Length; $i++) {
        $directory = $directories[$i]
        $commands = $directory_pyright_generate_typestubs_commands[$i]
        Set-Location $directory
        Write-Host "\t for $directory"
        foreach ($command in $commands) {
            Invoke-Expression $command
        }
        Write-Host "done."
    }
}

Export-PyrightTypestubs