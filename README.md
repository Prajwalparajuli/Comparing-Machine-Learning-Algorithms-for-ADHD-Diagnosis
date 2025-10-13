# Comparing Machine Learning Algorithms for ADHD Diagnosis Using Multimodal rs-fMRI and Clinical Features from the ADHD-200 Dataset

This repository compares SVM, Random Forest,  XGBoost, Multi modal DNN and Early Fusion LSTM for ADHD vs. Control classification using:
- Functional Connectivity (FC, CC200)
- Regional synchrony approximation (ROI-level ReHo-inspired)
- fALFF
- Selected phenotypic covariates (e.g., DX, Age, Sex, IQs, MedStatus, Site)

## Repository Layout
- **Feature Extraction jupyter notebooks/** — existing feature notebooks (kept as-is)
- **Machine Learning Models/** — existing modeling notebooks (kept as-is)
- **paper/** — manuscript and publication assets
  - **Comparing Machine Learning Algorithms for ADHD Diagnosis Using Multimodal rs-fMRI and Clinical Features from the ADHD-200 Dataset.pdf** (final manuscript)
  - **figures/** (moved from the original "Graphs and figures/" folder)
- **data/** — placeholder only; do NOT commit raw ADHD-200 data
  - **processed/** — where notebooks can save `X.npy`, `y.npy`, `sites.csv` if re-run
- **reports/** — placeholder for generated outputs (if notebooks are re-run)
  - **figures/** — plots (e.g., accuracy/F1 bar charts)
  - **tables/** — CSV summaries (e.g., cross-validation metrics)

## Quick Start (for future reproduction)
```bash
pip install -r requirements.txt
# Place required ADHD-200 inputs locally under data/ (not committed)
# Then open notebooks in:
#   "Feature Extraction jupyter notebooks/" to build features
#   "Machine Learning Models/" to train/evaluate models
