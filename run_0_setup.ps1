#!/usr/bin/env pwsh
# ===========================================================
#   Step 0: Environment Setup
#   Creates .venv and installs requirements.txt
#   Run this ONCE before training any models.
# ===========================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  ADHD ML Pipeline - Environment Setup" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

# --- Virtual Environment ---
if (-Not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "`n  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "  [OK] .venv created" -ForegroundColor Green
} else {
    Write-Host "`n  [OK] .venv already exists" -ForegroundColor Green
}

$PYTHON = ".venv\Scripts\python.exe"

# --- Dependencies ---
Write-Host "`n  Installing / updating dependencies..." -ForegroundColor Yellow
& $PYTHON -m pip install -q --upgrade pip
& $PYTHON -m pip install -q -r requirements.txt

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  Setup complete! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Next: .\run_1_train_fast.ps1   (Dummy + DecisionTree)" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
