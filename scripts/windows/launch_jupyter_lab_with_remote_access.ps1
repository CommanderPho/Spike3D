# Exit immediately if a command exits with a non-zero status
$ErrorActionPreference = "Stop"

# Get the script's directory and calculate project root
$SCRIPT_DIR = $PSScriptRoot
$PROJECT_DIR = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

# Variables
$JUPYTER_PORT = 8889
$JUPYTER_LOG = Join-Path $PROJECT_DIR "jupyter.log"  # Absolute path
$TIMEOUT = 60  # seconds to wait for Jupyter to start

# Function to copy text to clipboard
function Copy-ToClipboard {
    param([string]$text)
    Set-Clipboard -Value $text
}

# Navigate to your project directory
Write-Host "Navigating to project directory: $PROJECT_DIR"
Set-Location $PROJECT_DIR

# Ensure UV is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "UV could not be found. Please install UV first."
    exit 1
}

# Initialize cleanup variables
$jupyterProcess = $null
$job = $null

try {
    # Start Jupyter Lab in the background, redirecting output to a log file
    Write-Host "Starting Jupyter Lab..."
    
    # Use Start-Job to run in background with output redirection
    # This allows both stdout and stderr to be redirected to the same file
    $job = Start-Job -ScriptBlock {
        param($projectDir, $port, $logFile)
        Set-Location $projectDir
        & uv run jupyter-lab --no-browser --port=$port --NotebookApp.ip='0.0.0.0' --NotebookApp.allow_origin='*' *> $logFile
    } -ArgumentList $PROJECT_DIR, $JUPYTER_PORT, $JUPYTER_LOG
    
    Write-Host "Jupyter Lab started (Job ID: $($job.Id))"
    
    # Wait until Jupyter Lab is ready by checking the log file for the URL
    Write-Host "Waiting for Jupyter Lab to start and generate the URL..."
    $URL = ""
    $START_TIME = Get-Date
    
    while ($true) {
        if (Test-Path $JUPYTER_LOG) {
            # Attempt to extract the URL
            # Use -ErrorAction SilentlyContinue to handle file lock issues
            $logContent = Get-Content $JUPYTER_LOG -Raw -ErrorAction SilentlyContinue
            if ($logContent) {
                $match = [regex]::Match($logContent, "http://127\.0\.0\.1:$JUPYTER_PORT/lab\?token=\w+")
                if ($match.Success) {
                    $URL = $match.Value
                    Write-Host "Jupyter Lab URL found: $URL"
                    break
                }
            }
        }
        
        $CURRENT_TIME = Get-Date
        $ELAPSED = ($CURRENT_TIME - $START_TIME).TotalSeconds
        if ($ELAPSED -ge $TIMEOUT) {
            Write-Host "Timed out waiting for Jupyter Lab to start."
            Write-Host "Checking Jupyter Log for errors:"
            if (Test-Path $JUPYTER_LOG) {
                Get-Content $JUPYTER_LOG -ErrorAction SilentlyContinue
            }
            exit 1
        }
        
        Start-Sleep -Seconds 2
    }
    
    # Copy the URL to the clipboard
    if ($URL) {
        Write-Host "Copying Jupyter URL to clipboard..."
        Copy-ToClipboard -text $URL
        Write-Host "Jupyter URL copied to clipboard."
    } else {
        Write-Host "Failed to retrieve Jupyter Lab URL."
        exit 1
    }
    
    # Optionally, open the URL in the default browser
    # Uncomment the following line if you want to automatically open the browser
    # Start-Process $URL
    
    Write-Host "Jupyter Lab is running. Access it at: $URL"
    
    # Keep the script running to maintain Jupyter Lab
    # Wait for the job to complete
    Wait-Job $job | Out-Null
    Receive-Job $job | Out-Null
    
} finally {
    # Function to gracefully terminate Jupyter Lab on script exit
    if ($job) {
        Write-Host "Shutting down Jupyter Lab (Job ID: $($job.Id))..."
        try {
            Stop-Job $job -ErrorAction SilentlyContinue
            Remove-Job $job -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "Error shutting down Jupyter Lab job: $_"
        }
    }
}
