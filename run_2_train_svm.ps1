#!/usr/bin/env pwsh
# ===========================================================
#   Step 2: Train SVM_RBF (GridSearchCV)
#   SLOWEST model - expect 15-30+ min across all 6 datasets
#
#   Full run:      .\run_2_train_svm.ps1
#   One dataset:   .\run_2_train_svm.ps1 FC_PCA
#   Datasets: FC, FC_Pheno, FC_PCA, ROI_Sync, fALFF, Combined
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
Write-Host "  Training: SVM_RBF (GridSearchCV)" -ForegroundColor Cyan
Write-Host "  WARNING: This is the slowest model. Go grab a coffee." -ForegroundColor DarkGray
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& $PYTHON -m src.models.svm $args

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  SVM_RBF complete! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Results: results/SVM_RBF/" -ForegroundColor Green
Write-Host "  Next:    .\run_3_train_rf_xgb.ps1" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
