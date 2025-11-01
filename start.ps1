# !root/start.ps1
# --------------------------------------------------------------------------
# Author: twj0
# Project: heat-exchanger
# --------------------------------------------------------------------------

# 打印作者和项目信息
Write-Host "Author: twj0"
Write-Host "Project: heat-exchanger"
Write-Host ""

# 打印 ASCII Art
$asciiArt = @"
.__                   __                                 .__                                         
|  |__   ____ _____ _/  |_            ____ ___  ___ ____ |  |__ _____    ____    ____   ___________  
|  |  \_/ __ \\__  \\   __\  ______ _/ __ \\  \/  // ___\|  |  \\__  \  /    \  / ___\_/ __ \_  __ \ 
|   Y  \  ___/ / __ \|  |   /_____/ \  ___/ >    <\  \___|   Y  \/ __ \|   |  \/ /_/  >  ___/|  | \/ 
|___|  /\___  >____  /__|            \___  >__/\_ \\___  >___|  (____  /___|  /\___  / \___  >__|    
     \/     \/     \/                    \/      \/    \/     \/     \/     \//_____/      \/        
"@

Write-Host $asciiArt

$Script:RepoRoot = $PSScriptRoot
$Script:ProjectRoot = Join-Path $Script:RepoRoot 'project'
$Script:PythonExe = 'python'

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory)]
        [string]$RelativePath
    )

    if ([System.IO.Path]::IsPathRooted($RelativePath)) {
        return $RelativePath
    }

    return Join-Path $Script:ProjectRoot $RelativePath
}

function Show-GPUInfo {
    Write-Host "Checking GPU adapters..."
    try {
        $gpuAdapters = Get-CimInstance -ClassName Win32_VideoController |
            Select-Object Name, DriverVersion, (@{Name='DedicatedMemory(GB)';Expression={[Math]::Round($_.AdapterRAM / 1GB,2)}})

        if ($gpuAdapters -and $gpuAdapters.Count -gt 0) {
            $gpuAdapters | ForEach-Object {
                Write-Host "- Adapter: $($_.Name)" -ForegroundColor Green
                Write-Host "  Driver:  $($_.DriverVersion)"
                Write-Host "  VRAM:    $($_.'DedicatedMemory(GB)') GB"
            }
        } else {
            Write-Host "No discrete GPU adapters detected." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Unable to query GPU information: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Invoke-VenvActivation {
    $venvScript = Join-Path $Script:ProjectRoot '.venv\Scripts\Activate.ps1'
    if (Test-Path $venvScript) {
        try {
            Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Cyan
            . $venvScript
            return $true
        } catch {
            Write-Host "Virtual environment activation failed: $($_.Exception.Message)" -ForegroundColor Yellow
            return $false
        }
    }

    Write-Host "No virtual environment found at $venvScript. Continuing without activation." -ForegroundColor Yellow
    return $false
}

function Enter-ProjectLocation {
    param(
        [string]$RelativePath = '.'
    )

    $destination = Resolve-ProjectPath -RelativePath $RelativePath
    if (-not (Test-Path $destination)) {
        Write-Host "Target directory not found: $destination" -ForegroundColor Red
        return
    }

    try {
        Set-Location $destination
        Write-Host "Current working directory:" (Get-Location) -ForegroundColor Cyan
    } catch {
        Write-Host "Unable to change directory: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Show-NavigationMenu {
    Write-Host "" 
    Write-Host "Navigation targets:" -ForegroundColor Cyan
    Write-Host "  [1] project/"
    Write-Host "  [2] project/src/"
    Write-Host "  [3] project/data/"
    Write-Host "  [4] project/my_models/"
    Write-Host "  [5] project/logs/"
    Write-Host "  [6] project/configs/"
    Write-Host "  [7] project/docs/"
    Write-Host "  [8] project/simulate/"
    Write-Host "  [9] project/figure/"
    Write-Host "  [0] Back"

    $target = Read-Host "Select destination"
    switch ($target) {
        '1' { Enter-ProjectLocation '.' }
        '2' { Enter-ProjectLocation 'src' }
        '3' { Enter-ProjectLocation 'data' }
        '4' { Enter-ProjectLocation 'my_models' }
        '5' { Enter-ProjectLocation 'logs' }
        '6' { Enter-ProjectLocation 'configs' }
        '7' { Enter-ProjectLocation 'docs' }
        '8' { Enter-ProjectLocation 'simulate' }
        '9' { Enter-ProjectLocation 'figure' }
        default { Write-Host "Returning to main menu." -ForegroundColor Yellow }
    }
}

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory)]
        [string]$Script,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $Script:ProjectRoot
    )

    $python = Get-Command $Script:PythonExe -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Host "Python executable not found. Please ensure it is installed and on PATH." -ForegroundColor Red
        return
    }

    $scriptPath = Resolve-ProjectPath -RelativePath $Script
    if (-not (Test-Path $scriptPath)) {
        Write-Host "Python script not found: $scriptPath" -ForegroundColor Red
        return
    }

    $working = Resolve-ProjectPath -RelativePath $WorkingDirectory
    if (-not (Test-Path $working)) {
        Write-Host "Working directory not found: $working" -ForegroundColor Red
        return
    }

    Write-Host "" 
    Write-Host ">>> Executing: $Script:PythonExe $scriptPath $Arguments" -ForegroundColor Cyan

    Push-Location
    try {
        Set-Location $working
        & $Script:PythonExe $scriptPath @Arguments
    } finally {
        Pop-Location
    }
}

function Invoke-TrainingTask {
    param(
        [ValidateSet('PPO','SAC','DQN')]
        [string]$Algorithm = 'PPO',
        [string]$Config = 'configs/default.yaml',
        [int]$Timesteps = 200000,
        [string]$SavePath = 'my_models',
        [string]$LogPath = 'logs',
        [int]$Seed = 42
    )

    $arguments = @(
        '--config', $Config,
        '--algo', $Algorithm,
        '--timesteps', $Timesteps.ToString(),
        '--save-path', $SavePath,
        '--log-path', $LogPath,
        '--seed', $Seed.ToString()
    )

    Write-Host "Stable-Baselines3 ($Algorithm) training pipeline" -ForegroundColor Cyan
    Invoke-PythonScript -Script 'rl_algorithms/train.py' -Arguments $arguments
}

function Invoke-EvaluationTask {
    param(
        [string]$Config = 'configs/default.yaml',
        [string]$Baseline = 'simple_tou',
        [ValidateSet('PPO','SAC','DQN')]
        [string]$Algorithm = 'PPO',
        [string]$ModelPath,
        [int]$Episodes = 10,
        [string]$Output = 'results'
    )

    if ([string]::IsNullOrWhiteSpace($ModelPath)) {
        Write-Host "Model path is required for evaluation." -ForegroundColor Red
        return
    }

    $arguments = @(
        '--config', $Config,
        '--baseline', $Baseline,
        '--rl-model', $ModelPath,
        '--algo', $Algorithm,
        '--episodes', $Episodes.ToString(),
        '--output', $Output
    )

    Write-Host "Evaluation / inference workflow" -ForegroundColor Cyan
    Invoke-PythonScript -Script 'simulate/run_eval.py' -Arguments $arguments
}

function Invoke-DemoScenario {
    Invoke-PythonScript -Script 'demo.py'
}

function Test-UvAvailable {
    return [bool](Get-Command 'uv' -ErrorAction SilentlyContinue)
}

function Show-UvStatus {
    if (-not (Test-UvAvailable)) {
        Write-Host "uv is not available on PATH." -ForegroundColor Yellow
        return
    }

    Write-Host "uv version:" -ForegroundColor Cyan
    & uv --version
    Write-Host ""
    Write-Host "uv Python version:" -ForegroundColor Cyan
    & uv run python --version
}

function Invoke-UvSync {
    if (-not (Test-UvAvailable)) {
        Write-Host "uv is not available on PATH." -ForegroundColor Yellow
        return
    }

    Push-Location
    try {
        Set-Location $Script:ProjectRoot
        Write-Host "Running 'uv sync' in $Script:ProjectRoot" -ForegroundColor Cyan
        & uv sync
    } finally {
        Pop-Location
    }
}

function Show-UvPackages {
    if (-not (Test-UvAvailable)) {
        Write-Host "uv is not available on PATH." -ForegroundColor Yellow
        return
    }

    Write-Host "Installed packages (uv pip list):" -ForegroundColor Cyan
    Push-Location
    try {
        Set-Location $Script:ProjectRoot
        & uv pip list
    } finally {
        Pop-Location
    }
}

function Show-ProjectOverview {
    Write-Host "Project directories:" -ForegroundColor Cyan
    Get-ChildItem -Path $Script:ProjectRoot -Directory |
        Sort-Object Name |
        ForEach-Object { Write-Host "  $($_.Name)/" }
}

function Show-RecentModels {
    $modelsPath = Resolve-ProjectPath -RelativePath 'my_models'
    if (-not (Test-Path $modelsPath)) {
        Write-Host "Models directory not found." -ForegroundColor Yellow
        return
    }

    $artifacts = Get-ChildItem -Path $modelsPath -Filter '*.zip' -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 5

    if (-not $artifacts) {
        Write-Host "No model artifacts found." -ForegroundColor Yellow
        return
    }

    Write-Host "Recent model artifacts:" -ForegroundColor Cyan
    $artifacts | ForEach-Object {
        Write-Host "  $($_.LastWriteTime.ToString('yyyy-MM-dd HH:mm')) - $($_.FullName)"
    }
}

function Show-PythonInfo {
    $python = Get-Command $Script:PythonExe -ErrorAction SilentlyContinue
    if (-not $python) {
        Write-Host "Python executable not found." -ForegroundColor Yellow
        return
    }

    Write-Host "Python executable:" -ForegroundColor Cyan
    Write-Host "  $($python.Source)"
    Write-Host "Python version:" -ForegroundColor Cyan
    & $Script:PythonExe --version
}

function Show-UtilityMenu {
    Write-Host "" 
    Write-Host "General utilities:" -ForegroundColor Cyan
    Write-Host "  [1] Show GPU information"
    Write-Host "  [2] Show project directory overview"
    Write-Host "  [3] Show Python interpreter information"
    Write-Host "  [4] List recent model artifacts"
    Write-Host "  [0] Back"

    $choice = Read-Host "Select utility"
    switch ($choice) {
        '1' { Show-GPUInfo }
        '2' { Show-ProjectOverview }
        '3' { Show-PythonInfo }
        '4' { Show-RecentModels }
        default { Write-Host "Returning to main menu." -ForegroundColor Yellow }
    }
}

function Invoke-UvMenu {
    Write-Host "" 
    Write-Host "UV environment tools:" -ForegroundColor Cyan
    Write-Host "  [1] Show uv status"
    Write-Host "  [2] Run 'uv sync'"
    Write-Host "  [3] List packages via 'uv pip list'"
    Write-Host "  [0] Back"

    $choice = Read-Host "Select UV action"
    switch ($choice) {
        '1' { Show-UvStatus }
        '2' { Invoke-UvSync }
        '3' { Show-UvPackages }
        default { Write-Host "Returning to main menu." -ForegroundColor Yellow }
    }
}

function Invoke-Workflow {
    param(
        [Parameter(Mandatory)]
        [string]$Selection
    )

    switch ($Selection) {
        '1' {
            Show-NavigationMenu
        }
        '2' {
            $scriptInput = Read-Host "Python script path (relative to project)"
            if ([string]::IsNullOrWhiteSpace($scriptInput)) {
                Write-Host "Script path is required." -ForegroundColor Yellow
                break
            }
            $argsInput = Read-Host "Arguments (optional, space-separated)"
            $scriptArgs = @()
            if (-not [string]::IsNullOrWhiteSpace($argsInput)) {
                $scriptArgs = $argsInput -split '\s+'
            }
            $workingInput = Read-Host "Working directory (relative, default: project root)"
            if ([string]::IsNullOrWhiteSpace($workingInput)) {
                $workingInput = '.'
            }
            Invoke-PythonScript -Script $scriptInput -Arguments $scriptArgs -WorkingDirectory $workingInput
        }
        '3' {
            $algoInput = Read-Host "Algorithm [PPO/SAC/DQN] (default: PPO)"
            if ([string]::IsNullOrWhiteSpace($algoInput)) {
                $algoInput = 'PPO'
            }
            $algoInput = $algoInput.ToUpper()
            if ('PPO','SAC','DQN' -notcontains $algoInput) {
                Write-Host "Unsupported algorithm." -ForegroundColor Red
                break
            }
            $configInput = Read-Host "Config path (default: configs/default.yaml)"
            if ([string]::IsNullOrWhiteSpace($configInput)) {
                $configInput = 'configs/default.yaml'
            }
            $timestepsInput = Read-Host "Training timesteps (default: 200000)"
            if ([string]::IsNullOrWhiteSpace($timestepsInput)) {
                $timestepsInput = '200000'
            }
            $saveInput = Read-Host "Model output directory (default: my_models)"
            if ([string]::IsNullOrWhiteSpace($saveInput)) {
                $saveInput = 'my_models'
            }
            $logInput = Read-Host "Log directory (default: logs)"
            if ([string]::IsNullOrWhiteSpace($logInput)) {
                $logInput = 'logs'
            }
            $seedInput = Read-Host "Random seed (default: 42)"
            if ([string]::IsNullOrWhiteSpace($seedInput)) {
                $seedInput = '42'
            }

            try {
                Invoke-TrainingTask -Algorithm $algoInput -Config $configInput -Timesteps ([int]$timestepsInput) -SavePath $saveInput -LogPath $logInput -Seed ([int]$seedInput)
            } catch {
                Write-Host "Training task failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
        '4' {
            $algoInput = Read-Host "Algorithm [PPO/SAC/DQN] (default: PPO)"
            if ([string]::IsNullOrWhiteSpace($algoInput)) {
                $algoInput = 'PPO'
            }
            $algoInput = $algoInput.ToUpper()
            if ('PPO','SAC','DQN' -notcontains $algoInput) {
                Write-Host "Unsupported algorithm." -ForegroundColor Red
                break
            }
            $configInput = Read-Host "Config path (default: configs/default.yaml)"
            if ([string]::IsNullOrWhiteSpace($configInput)) {
                $configInput = 'configs/default.yaml'
            }
            $modelInput = Read-Host "Model path (e.g., my_models/${algoInput}_latest_final.zip)"
            if ([string]::IsNullOrWhiteSpace($modelInput)) {
                Write-Host "Model path is required." -ForegroundColor Yellow
                break
            }
            $episodesInput = Read-Host "Number of evaluation episodes (default: 10)"
            if ([string]::IsNullOrWhiteSpace($episodesInput)) {
                $episodesInput = '10'
            }
            $baselineInput = Read-Host "Baseline controller (default: simple_tou)"
            if ([string]::IsNullOrWhiteSpace($baselineInput)) {
                $baselineInput = 'simple_tou'
            }
            $outputInput = Read-Host "Output directory (default: results)"
            if ([string]::IsNullOrWhiteSpace($outputInput)) {
                $outputInput = 'results'
            }

            try {
                Invoke-EvaluationTask -Config $configInput -Baseline $baselineInput -Algorithm $algoInput -ModelPath $modelInput -Episodes ([int]$episodesInput) -Output $outputInput
            } catch {
                Write-Host "Evaluation task failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
        '5' {
            Invoke-DemoScenario
        }
        '6' {
            Invoke-UvMenu
        }
        '7' {
            Show-UtilityMenu
        }
        '0' {
            Write-Host "Exiting launcher."
        }
        default {
            Write-Host "Invalid selection. Please choose a valid option." -ForegroundColor Yellow
        }
    }
}

# --------------------------------------------------------------------------
# GPU 信息检测（不依赖 Python 环境）
# --------------------------------------------------------------------------
Show-GPUInfo

# --------------------------------------------------------------------------
# 导航至项目根目录
# --------------------------------------------------------------------------
$projectRoot = $Script:ProjectRoot
if (-not (Test-Path $projectRoot)) {
    Write-Host "Project directory not found: $projectRoot" -ForegroundColor Red
    exit 1
}

Enter-ProjectLocation '.'

$activated = Invoke-VenvActivation
if (-not $activated) {
    Write-Host "Tip: Use Invoke-VenvActivation after creating a virtual environment." -ForegroundColor Yellow
}

# --------------------------------------------------------------------------
# 用户自定义优化程序入口
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "Available optimization workflows:" -ForegroundColor Cyan
Write-Host "  [1] Navigate between project directories"
Write-Host "  [2] Execute Python script"
Write-Host "  [3] Train RL agent (Stable-Baselines3 / PyTorch)"
Write-Host "  [4] Evaluate or run inference"
Write-Host "  [5] Run demo scenarios"
Write-Host "  [6] UV environment tools"
Write-Host "  [7] General utilities"
Write-Host "  [0] Quit"

do {
    $currentPath = Get-Location
    if ($currentPath -ne $projectRoot) {
        try {
            Set-Location $projectRoot
            Write-Host "Switched working directory to $projectRoot" -ForegroundColor Cyan
        } catch {
            Write-Host "Unable to change directory to $projectRoot $($_.Exception.Message)" -ForegroundColor Red
            break
        }
    }

    $selection = Read-Host "Select workflow (0-7)"
    $selection = $selection.ToUpper()
    if ($selection -eq '0') {
        Invoke-Workflow -Selection '0'
        break
    }
    elseif ('1','2','3','4','5','6','7' -contains $selection) {
        try {
            Invoke-Workflow -Selection $selection
        } catch {
            Write-Host "Workflow failed: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "Invalid selection. Please choose 1, 2, 3, or 0." -ForegroundColor Yellow
    }
} while ($true)