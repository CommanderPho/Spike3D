$directories = @(
    "C:\Users\pho\repos\Spike3DWorkEnv\NeuroPy",
    "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoCoreHelpers",
    "C:\Users\pho\repos\Spike3DWorkEnv\pyPhoPlaceCellAnalysis",
    "C:\Users\pho\repos\Spike3DWorkEnv\Spike3D"
)

$directory_commands = @(
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt", "pyright --createstub neuropy"), # , "another_command", "yet_another_command"
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt", "pyright --createstub pyphocorehelpers"), # , "command_two", "command_three"
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt", "pyright --createstub pyphoplacecellanalysis"), # , "command_four", "command_five"
    @("poetry export --without-hashes --format=requirements.txt  > requirements.txt", "pyright --createstub neuropy", "pyright --createstub pyphocorehelpers", "pyright --createstub pyphoplacecellanalysis")
)

for ($i = 0; $i -lt $directories.Length; $i++) {
    $directory = $directories[$i]
    $commands = $directory_commands[$i]
    Set-Location $directory
    foreach ($command in $commands) {
        Invoke-Expression $command
    }
}
