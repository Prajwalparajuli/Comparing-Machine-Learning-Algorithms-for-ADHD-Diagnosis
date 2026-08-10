#!/usr/bin/env pwsh
# ===========================================================
#   Step 4: Train DNN_TF (TensorFlow deep neural network)
#   Requires TensorFlow - skip this if not installed
#   ~5-15 min total
#
#   Full run:      .\run_4_train_dnn.ps1
#   One dataset:   .\run_4_train_dnn.ps1 FC_PCA
#   Datasets: FC, FC_Pheno, FC_PCA, ROI_Sync, fALFF, Combined
# ===========================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$env:PYTHONUNBUFFERED = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "3"

$PYTHON = ".venv\Scripts\python.exe"
if (-Not (Test-Path $PYTHON)) {
    Write-Host "  [ERROR] .venv not found. Run .\run_0_setup.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Training: DNN_TF (TensorFlow)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

# Check TensorFlow availability before running
$tfCheck = & $PYTHON -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow; print('OK')" 2>$null

if ($tfCheck -notmatch "OK") {
    Write-Host "`n  [SKIP] TensorFlow not installed." -ForegroundColor DarkYellow
    Write-Host "  Install with: .venv\Scripts\pip.exe install tensorflow" -ForegroundColor DarkGray
    Write-Host "  You can proceed to .\run_5_evaluate.ps1 without DNN." -ForegroundColor DarkGray
    exit 0
}

& $PYTHON -m src.models.dnn_tf $args

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  DNN_TF complete! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Results: results/DNN_TF/" -ForegroundColor Green
Write-Host "  Next:    .\run_5_evaluate.ps1" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
