#!/usr/bin/env pwsh
# ===========================================================
#   ADHD ML Comparison Pipeline - Full Orchestrator
# ===========================================================
#
#  Runs ALL steps end-to-end. If you want to run step-by-step,
#  use the individual scripts instead:
#
#  .\run_0_setup.ps1           Environment + dependencies
#  .\run_1_train_fast.ps1      DummyBaseline + DecisionTree  (~2-5 min)
#  .\run_2_train_svm.ps1       SVM_RBF                       (~15-30 min)
#  .\run_3_train_rf_xgb.ps1    RandomForest + XGBoost         (~10-20 min)
#  .\run_4_train_dnn.ps1       DNN_TF (needs TensorFlow)      (~5-15 min)
#  .\run_5_evaluate.ps1        Bootstrap CI + McNemar tests    (~1-2 min)
#  .\run_6_visualize.ps1       Publication-quality charts      (~30s)
#
#  Each training script also accepts dataset filters:
#    .\run_2_train_svm.ps1 FC_PCA Combined
#
#  Datasets: FC, FC_Pheno, FC_PCA, ROI_Sync, fALFF, Combined
#
# ===========================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  ADHD ML Comparison Pipeline - Full Run" -ForegroundColor Cyan
Write-Host "  5 Models x 6 Datasets = 30 experiment cells" -ForegroundColor Cyan
Write-Host "  WARNING: This will take 30-90 min. Use individual" -ForegroundColor DarkYellow
Write-Host "    scripts if you want to run step-by-step." -ForegroundColor DarkYellow
Write-Host "===========================================================" -ForegroundColor Cyan

$sw = [System.Diagnostics.Stopwatch]::StartNew()

& "$PSScriptRoot\run_0_setup.ps1"
& "$PSScriptRoot\run_1_train_fast.ps1" $args
& "$PSScriptRoot\run_2_train_svm.ps1" $args
& "$PSScriptRoot\run_3_train_rf_xgb.ps1" $args
& "$PSScriptRoot\run_4_train_dnn.ps1" $args
& "$PSScriptRoot\run_5_evaluate.ps1"
& "$PSScriptRoot\run_6_visualize.ps1"

$sw.Stop()
Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  FULL PIPELINE COMPLETE! ($([math]::Round($sw.Elapsed.TotalMinutes, 1)) min)" -ForegroundColor Green
Write-Host "  Results:   results/summary_all.csv" -ForegroundColor Green
Write-Host "  Charts:    results/charts/" -ForegroundColor Green
Write-Host "  Per-cell:  results/{Model}/{Dataset}/" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
