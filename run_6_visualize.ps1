#!/usr/bin/env pwsh
# ===========================================================
#   Step 6: Generate Visualizations
#   Publication-quality charts from existing results
#   ~30s (just plotting, no computation)
#
#   Requires: run_5_evaluate.ps1 first (needs bootstrap_ci.csv)
# ===========================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PYTHON = ".venv\Scripts\python.exe"
if (-Not (Test-Path $PYTHON)) {
    Write-Host "  [ERROR] .venv not found. Run .\run_0_setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Generating Publication-Quality Charts" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& $PYTHON -m src.visualize

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  Charts complete! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Output: results/charts/" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
