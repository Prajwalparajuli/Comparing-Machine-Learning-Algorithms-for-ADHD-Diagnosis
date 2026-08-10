#!/usr/bin/env pwsh
# ===========================================================
#   Step 1: Train FAST models - DummyBaseline + DecisionTree
#   ~2-5 min total (both are lightweight)
#
#   Full run:      .\run_1_train_fast.ps1
#   One dataset:   .\run_1_train_fast.ps1 FC_PCA
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
Write-Host "  Training: DummyBaseline + DecisionTree" -ForegroundColor Cyan
Write-Host "  (Fast models - ~2-5 min total)" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "`n  [1/2] DummyBaseline..." -ForegroundColor Yellow
& $PYTHON -m src.models.dummy_baseline $args

Write-Host "`n  [2/2] DecisionTree (GridSearchCV)..." -ForegroundColor Yellow
& $PYTHON -m src.models.decision_tree $args

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  Done! ($([math]::Round($sw.Elapsed.TotalSeconds))s)" -ForegroundColor Green
Write-Host "  Results: results/DummyBaseline/ and results/DecisionTree/" -ForegroundColor Green
Write-Host "  Next:    .\run_2_train_svm.ps1" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
