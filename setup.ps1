# nfl-predict/setup.ps1
# Automated setup script for Windows PowerShell

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "NFL 2025 Season Prediction Model - Setup" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}

# Check Python version
$versionNumber = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$versionNumber -lt [version]"3.9") {
    Write-Host "✗ Python 3.9 or higher required. Found: $versionNumber" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python version is compatible" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
} else {
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "⚠ Could not activate automatically. Please run: .\venv\Scripts\Activate" -ForegroundColor Yellow
}
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "✓ pip upgraded" -ForegroundColor Green
Write-Host ""

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ All dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to install some dependencies" -ForegroundColor Red
    Write-Host "Try running manually: pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# Create necessary directories (if not exist)
Write-Host "Setting up directory structure..." -ForegroundColor Yellow
$directories = @(
    "data\raw",
    "data\processed",
    "outputs\figures",
    "outputs\reports",
    "models",
    "notebooks"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✓ Created: $dir" -ForegroundColor Green
    }
}
Write-Host "✓ Directory structure ready" -ForegroundColor Green
Write-Host ""

# Verify installation
Write-Host "Verifying installation..." -ForegroundColor Yellow
$testImports = @(
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "nflreadpy"
)

$allGood = $true
foreach ($module in $testImports) {
    try {
        python -c "import $module" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $module" -ForegroundColor Green
        } else {
            Write-Host "✗ $module" -ForegroundColor Red
            $allGood = $false
        }
    } catch {
        Write-Host "✗ $module" -ForegroundColor Red
        $allGood = $false
    }
}
Write-Host ""

# Final status
Write-Host "===============================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✓ SETUP COMPLETE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Run the analysis: python main.py" -ForegroundColor White
    Write-Host "2. Or explore with Jupyter: jupyter notebook" -ForegroundColor White
    Write-Host "3. See QUICKSTART.md for more options" -ForegroundColor White
} else {
    Write-Host "⚠ SETUP COMPLETED WITH WARNINGS" -ForegroundColor Yellow
    Write-Host "Some packages may need manual installation." -ForegroundColor Yellow
    Write-Host "Try: pip install -r requirements.txt" -ForegroundColor White
}
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

