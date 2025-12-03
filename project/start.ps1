# ============================================================================
# Carbon-Aware Building Control - Quick Start Script
# ============================================================================
# 
# Usage: Right-click and "Run with PowerShell" or execute in terminal:
#   .\start.ps1
#
# Author: Auto-generated for Applied Energy publication
# ============================================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

# Colors for output
function Write-Color {
    param([string]$Text, [string]$Color = "White")
    Write-Host $Text -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Color ("=" * 60) "Cyan"
    Write-Color "  $Title" "Cyan"
    Write-Color ("=" * 60) "Cyan"
    Write-Host ""
}

function Write-Success { param([string]$Text) Write-Color "[OK] $Text" "Green" }
function Write-Error { param([string]$Text) Write-Color "[ERROR] $Text" "Red" }
function Write-Info { param([string]$Text) Write-Color "[INFO] $Text" "Yellow" }

# ============================================================================
# Menu Functions
# ============================================================================

function Show-Menu {
    Clear-Host
    Write-Header "Carbon-Aware Building Control"
    Write-Color "  Project: Deep RL for TES Optimization" "White"
    Write-Color "  Location: $ProjectRoot" "DarkGray"
    Write-Host ""
    Write-Color ("-" * 60) "DarkGray"
    Write-Host ""
    
    Write-Color "  [1] Quick Training Test (5k steps)" "White"
    Write-Color "  [2] Full Training (500k steps)" "White"
    Write-Color "  [3] Evaluate Baselines" "White"
    Write-Color "  [4] Run Quick Experiment" "White"
    Write-Color "  [5] Run Full Experiment" "White"
    Write-Host ""
    Write-Color "  [6] Check EnergyPlus Setup" "Cyan"
    Write-Color "  [7] Run EnergyPlus Simulation" "Cyan"
    Write-Color "  [8] Start TensorBoard" "Cyan"
    Write-Host ""
    Write-Color "  [9] Open Project in VS Code" "Magenta"
    Write-Color "  [0] Exit" "DarkGray"
    Write-Host ""
    Write-Color ("-" * 60) "DarkGray"
}

function Enable-Venv {
    $venvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $venvPath) {
        & $venvPath
        return $true
    } else {
        Write-Error "Virtual environment not found at $venvPath"
        return $false
    }
}

function Invoke-PythonScript {
    param([string]$Script, [string[]]$Arguments)
    
    $pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $pythonPath)) {
        Write-Error "Python not found. Please create virtual environment first."
        return
    }
    
    $scriptPath = Join-Path $ProjectRoot $Script
    Write-Info "Running: python $Script $($Arguments -join ' ')"
    Write-Host ""
    
    & $pythonPath $scriptPath @Arguments
}

# ============================================================================
# Action Functions
# ============================================================================

function Start-QuickTraining {
    Write-Header "Quick Training Test"
    Invoke-PythonScript "train_rl.py" @("--timesteps", "5000", "--eval-freq", "2500")
    Write-Host ""
    Write-Info "Quick training completed!"
    Pause
}

function Start-FullTraining {
    Write-Header "Full Training (500k steps)"
    Write-Info "This will take approximately 1-2 hours..."
    Write-Host ""
    
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -eq "y") {
        Invoke-PythonScript "train_rl.py" @("--timesteps", "500000")
    }
    Pause
}

function Start-BaselineEvaluation {
    Write-Header "Baseline Evaluation"
    Invoke-PythonScript "evaluate_rl.py" @("--quick", "--episodes", "10")
    Pause
}

function Start-QuickExperiment {
    Write-Header "Quick Experiment"
    Invoke-PythonScript "run_experiments.py" @("--quick")
    Pause
}

function Start-FullExperiment {
    Write-Header "Full Experiment"
    Write-Info "This will train multiple models and generate all results..."
    Write-Host ""
    
    $confirm = Read-Host "Continue? (y/n)"
    if ($confirm -eq "y") {
        Invoke-PythonScript "run_experiments.py" @("--all", "--timesteps", "100000", "--pareto-timesteps", "50000")
    }
    Pause
}

function Test-EnergyPlus {
    Write-Header "EnergyPlus Setup Check"
    Invoke-PythonScript "envs\real_eplus_env.py"
    Pause
}

function Start-EnergyPlusSimulation {
    Write-Header "EnergyPlus Simulation"
    
    $idfPath = Join-Path $ProjectRoot "outputs\sim_building.idf"
    $weatherPath = Join-Path $ProjectRoot "data\weather\Shanghai_2024.epw"
    
    Write-Info "IDF: $idfPath"
    Write-Info "Weather: $weatherPath"
    Write-Host ""
    
    # Check if EnergyPlus is available
    $eplusDir = $env:ENERGYPLUS_DIR
    if (-not $eplusDir) {
        # Try common paths (including D: drive and various versions)
        $commonPaths = @(
            "D:\energyplus\2320",
            "D:\EnergyPlus\2320",
            "D:\energyplus",
            "C:\EnergyPlusV24-2-0",
            "C:\EnergyPlusV24-1-0",
            "C:\EnergyPlusV23-2-0",
            "C:\EnergyPlusV23-1-0",
            "C:\EnergyPlusV22-2-0"
        )
        foreach ($path in $commonPaths) {
            if (Test-Path $path) {
                $eplusDir = $path
                Write-Info "Found EnergyPlus at: $path"
                break
            }
        }
        
        # Also try to find via where command
        if (-not $eplusDir) {
            try {
                $whereResult = where.exe energyplus 2>$null
                if ($whereResult) {
                    $eplusDir = Split-Path $whereResult -Parent
                    Write-Info "Found EnergyPlus via PATH: $eplusDir"
                }
            } catch {}
        }
    }
    
    if ($eplusDir -and (Test-Path $eplusDir)) {
        $eplusExe = Join-Path $eplusDir "energyplus.exe"
        $outputDir = Join-Path $ProjectRoot "outputs\manual_run"
        
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        
        Write-Info "Running EnergyPlus..."
        & $eplusExe -w $weatherPath -d $outputDir $idfPath
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Simulation completed!"
            Write-Info "Results in: $outputDir"
        } else {
            Write-Error "Simulation failed with code $LASTEXITCODE"
        }
    } else {
        Write-Error "EnergyPlus not found!"
        Write-Info "Please install EnergyPlus and set ENERGYPLUS_DIR environment variable"
        Write-Info "Download: https://energyplus.net/downloads"
    }
    Pause
}

function Start-TensorBoard {
    Write-Header "TensorBoard"
    
    $logDir = Join-Path $ProjectRoot "outputs\logs"
    Write-Info "Log directory: $logDir"
    Write-Info "Opening TensorBoard at http://localhost:6006"
    Write-Host ""
    
    $pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    Start-Process $pythonPath -ArgumentList "-m", "tensorboard.main", "--logdir=$logDir"
    
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:6006"
    
    Write-Success "TensorBoard started!"
    Pause
}

function Open-VSCode {
    Write-Info "Opening VS Code..."
    code $ProjectRoot
}

# ============================================================================
# Utility Functions
# ============================================================================

function Show-ProjectStatus {
    Write-Header "Project Status"
    
    # Check virtual environment
    $venvPath = Join-Path $ProjectRoot ".venv"
    if (Test-Path $venvPath) {
        Write-Success "Virtual environment exists"
    } else {
        Write-Error "Virtual environment not found"
    }
    
    # Check models
    $modelsDir = Join-Path $ProjectRoot "outputs\models"
    if (Test-Path $modelsDir) {
        $modelCount = (Get-ChildItem $modelsDir -Directory).Count
        Write-Info "Trained models: $modelCount"
    }
    
    # Check results
    $resultsDir = Join-Path $ProjectRoot "outputs\results"
    if (Test-Path $resultsDir) {
        $resultFiles = (Get-ChildItem $resultsDir -File).Count
        Write-Info "Result files: $resultFiles"
    }
    
    Pause
}

function Install-Dependencies {
    Write-Header "Installing Dependencies"
    
    $pythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $pipPath = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
    $reqPath = Join-Path $ProjectRoot "requirements.txt"
    
    if (Test-Path $pipPath) {
        Write-Info "Installing from requirements.txt..."
        & $pipPath install -r $reqPath
        Write-Success "Dependencies installed!"
    } else {
        Write-Error "pip not found in virtual environment"
    }
    Pause
}

# ============================================================================
# Main Loop
# ============================================================================

function Main {
    while ($true) {
        Show-Menu
        $choice = Read-Host "Select option"
        
        switch ($choice) {
            "1" { Start-QuickTraining }
            "2" { Start-FullTraining }
            "3" { Start-BaselineEvaluation }
            "4" { Start-QuickExperiment }
            "5" { Start-FullExperiment }
            "6" { Test-EnergyPlus }
            "7" { Start-EnergyPlusSimulation }
            "8" { Start-TensorBoard }
            "9" { Open-VSCode; break }
            "0" { 
                Write-Color "Goodbye!" "Green"
                exit 
            }
            "status" { Show-ProjectStatus }
            "install" { Install-Dependencies }
            default { Write-Error "Invalid option: $choice" }
        }
    }
}

# ============================================================================
# Entry Point
# ============================================================================

# Check if running from correct directory
if (-not (Test-Path (Join-Path $ProjectRoot "train_rl.py"))) {
    Write-Error "Please run this script from the project directory!"
    exit 1
}

# Run main menu
Main
