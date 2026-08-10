#!/usr/bin/env pwsh
# ===========================================================
#   Step 3: Train RandomForest + XGBoost
#   ~10-20 min total (both use GridSearchCV but faster than SVM)
#
#   Full run:      .\run_3_train_rf_xgb.ps1
#   One dataset:   .\run_3_train_rf_xgb.ps1 FC_PCA
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
Write-Host "  Training: RandomForest + XGBoost" -ForegroundColor Cyan
Write-Host "  (~10-20 min total)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "`n  [1/2] RandomForest (GridSearchCV)..." -ForegroundColor Yellow
& $PYTHON -m src.models.random_forest $args

Write-Host "`n  [2/2] XGBoost (GridSearchCV)..." -ForegroundColor Yellow
& $PYTHON -m src.models.xgboost_model $args

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  RF + XGBoost complete! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Results: results/RandomForest/ and results/XGBoost/" -ForegroundColor Green
Write-Host "  Next:    .\run_4_train_dnn.ps1  (if TF installed)" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
