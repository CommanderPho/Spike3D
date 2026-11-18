# Exit immediately if a command exits with a non-zero status
$ErrorActionPreference = "Stop"

# Get the script's directory and calculate project root
$SCRIPT_DIR = $PSScriptRoot
$PROJECT_DIR = Split-Path -Parent (Split-Path -Parent $SCRIPT_DIR)

# Variables
$JUPYTER_PORT = 8889
$JUPYTER_LOG = Join-Path $PROJECT_DIR "jupyter.log"  # Absolute path
$TIMEOUT = 60  # seconds to wait for Jupyter to start
$PRINT_JUPYTER_LOG = $true  # stream JUPYTER_LOG contents to console by default

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

# Process handle for cleanup (kill process tree on exit)
$proc = $null
$script:shutdownRequested = $false

# Ctrl+C: request shutdown so wait loop exits and finally runs (process tree kill)
try {
    [Console]::CancelKeyPress.Add({
        param($sender, $e)
        $script:shutdownRequested = $true
        $e.Cancel = $true
    })
} catch { }

try {
    # Start Jupyter Lab in a child process with stdout/stderr redirected to log
    Write-Host "Starting Jupyter Lab..."
    $childCmd = "Set-Location '$PROJECT_DIR'; uv run jupyter-lab --no-browser --port=$JUPYTER_PORT --ServerApp.ip='0.0.0.0' --ServerApp.allow_origin='*' --ServerApp.disable_check_xsrf=True *> '$JUPYTER_LOG'"
    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-Command", $childCmd -WorkingDirectory $PROJECT_DIR -NoNewWindow -PassThru
    Write-Host "Jupyter Lab started (PID: $($proc.Id))"

    # Poll log file until URL appears or timeout
    Write-Host "Waiting for Jupyter Lab to start and generate the URL..."
    $URL = ""
    $START_TIME = Get-Date
    while ($true) {
        if (Test-Path $JUPYTER_LOG) {
            $logContent = Get-Content $JUPYTER_LOG -Raw -ErrorAction SilentlyContinue
            if ($logContent) {
                $match = [regex]::Match($logContent, 'http://127\.0\.0\.1:\d+/lab\?token=\w+')
                if ($match.Success) {
                    $URL = $match.Value
                    Write-Host "Jupyter Lab URL found: $URL"
                    break
                }
            }
        }
        $ELAPSED = ((Get-Date) - $START_TIME).TotalSeconds
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

    # Copy URL to clipboard and print
    if ($URL) {
        Write-Host "Copying Jupyter URL to clipboard..."
        Copy-ToClipboard -text $URL
        Write-Host "Jupyter URL copied to clipboard."
    } else {
        Write-Host "Failed to retrieve Jupyter Lab URL."
        exit 1
    }
    Write-Host "Jupyter Lab is running. Access it at: $URL"

    # Wait loop: keep running and optionally stream new log lines to console
    $lastLineCount = 0
    while (-not $proc.HasExited -and -not $script:shutdownRequested) {
        if ($PRINT_JUPYTER_LOG -and (Test-Path $JUPYTER_LOG)) {
            $lines = @(Get-Content $JUPYTER_LOG -ErrorAction SilentlyContinue)
            if ($lines.Count -gt $lastLineCount) {
                $newLines = $lines[$lastLineCount..($lines.Count - 1)]
                foreach ($line in $newLines) { Write-Host $line }
                $lastLineCount = $lines.Count
            }
        }
        Start-Sleep -Milliseconds 300
    }
} finally {
    if ($proc) {
        try {
            if (-not $proc.HasExited) {
                Write-Host "Shutting down Jupyter Lab (PID: $($proc.Id))..."
                & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
            }
        } catch {
            Write-Host "Error shutting down Jupyter Lab: $_"
        }
    }
}
