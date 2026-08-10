#!/usr/bin/env pwsh
# ===========================================================
#   Step 5: Statistical Evaluation
#   Bootstrap CIs + McNemar's tests on all existing results
#   ~1-2 min (runs on saved OOF predictions, no retraining)
#
#   You can run this after ANY training step - it evaluates
#   whatever model results exist in results/
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
Write-Host "  Statistical Evaluation (Bootstrap CI + McNemar)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& $PYTHON -m src.evaluate

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  Evaluation complete! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Output: results/bootstrap_ci.csv" -ForegroundColor Green
Write-Host "          results/mcnemar_tests.csv" -ForegroundColor Green
Write-Host "  Next:   .\run_6_visualize.ps1" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
