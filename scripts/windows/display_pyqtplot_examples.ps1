# Display a start message
Write-Host "Running display_pyqtplot_examples.ps1..." -ForegroundColor Green

try {
    # Get the repository root path - adjust if needed based on your folder structure
    $repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    
    # Define paths
    $pythonPath = Join-Path -Path $repoRoot -ChildPath ".venv\Scripts\python.exe"
    $scriptPath = Join-Path -Path $repoRoot -ChildPath "LibrariesExamples\PyQtPlot\PyQtPlot_EXAMPLES.py"
    
    # Verify paths exist
    if (-not (Test-Path $pythonPath)) {
        Write-Error "Python interpreter not found at: $pythonPath"
        exit 1
    }
    
    if (-not (Test-Path $scriptPath)) {
        Write-Error "PyQtPlot examples script not found at: $scriptPath"
        exit 1
    }
    
    # Execute the Python script
    Write-Host "Executing: $pythonPath $scriptPath" -ForegroundColor Cyan
    & $pythonPath $scriptPath
    
    # Check if execution was successful
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Script executed successfully!" -ForegroundColor Green
    } else {
        Write-Host "Script execution failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Error "An error occurred: $_"
    exit 1
} finally {
    Write-Host "Done." -ForegroundColor Green
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
