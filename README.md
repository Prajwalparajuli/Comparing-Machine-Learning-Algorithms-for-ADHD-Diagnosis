# Comparing Machine Learning Algorithms for ADHD Diagnosis Using Multimodal rs-fMRI and Clinical Features from the ADHD-200 Dataset

[![IEEE Paper PDF](https://img.shields.io/badge/Paper-IEEE%20Format%20PDF-blue)](paper/Comparing%20Machine%20Learning%20Algorithms%20for%20ADHD%20Diagnosis%20Using%20Multimodal%20rs-fMRI%20and%20Clinical%20Features%20from%20the%20ADHD-200%20Dataset.pdf)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete reproducible pipeline, LaTeX manuscript, and evaluation benchmark comparing five machine learning algorithms (**Support Vector Machines**, **Decision Trees**, **Random Forests**, **XGBoost**, and a **Deep Neural Network**) for Attention-Deficit/Hyperactivity Disorder (ADHD) versus Control classification using resting-state fMRI (rs-fMRI) and clinical metadata from the **ADHD-200** dataset.

---

## Executive Summary

- **Top Model**: **XGBoost** trained on Functional Connectivity (FC) + Phenotypic features achieved **76.6% Accuracy** (95% CI: 72.8%–80.3%), **0.757 F1-Macro** (95% CI: 0.719–0.794), and **0.819 ROC-AUC** (95% CI: 0.779–0.857).
- **Statistical Significance**: Pairwise McNemar's tests confirmed XGBoost significantly outperformed all other models on multimodal inputs ($p < 0.001$ vs. DNN, Random Forest, SVM, and Baseline; $p < 0.01$ vs. Decision Tree).
- **Imaging-Only Baseline**: On imaging features alone (FC, fALFF, ROI Synchrony), models reached ~62% top accuracy (SVM on full FC: 62.3%), consistent with published ADHD-200 competition benchmarks.
- **Multimodal Synergy**: Adding non-leaky clinical metadata (IQ measures, age, sex, medication status) to imaging features improved XGBoost classification accuracy by **+14.8 percentage points** (61.8% $\rightarrow$ 76.6%).
- **Methodological Rigor**: All models were evaluated under strict stratified 10-fold cross-validation with no data leakage (circular diagnostic subscale scores excluded; scaling and PCA fitted strictly inside training folds).

---

## Performance Summary Table

All metrics are derived from Out-of-Fold (OOF) predictions across the 620-subject dataset using stratified 10-fold cross-validation. 95% confidence intervals are calculated via 2,000 bootstrap iterations.

| Feature Configuration | Best Model | Accuracy (95% CI) | F1-Macro (95% CI) | ROC-AUC (95% CI) | Key Finding |
|---|---|---|---|---|---|
| **FC + Phenotypic** | **XGBoost** | **76.6% [72.8%, 80.3%]** | **0.757 [0.719, 0.794]** | **0.819 [0.779, 0.857]** | **Top Multimodal Model** |
| **FC (Full 17,955)** | **SVM (RBF)** | **62.3% [58.5%, 66.3%]** | **0.617 [0.579, 0.655]** | **0.646 [0.603, 0.689]** | Top Imaging-Only Model |
| **fALFF (190 ROIs)** | **XGBoost** | **61.7% [57.9%, 65.5%]** | **0.592 [0.553, 0.630]** | **0.629 [0.584, 0.672]** | Local Spectral Metric |
| **Combined (PCA+Sync+fALFF)** | **XGBoost** | **61.0% [57.1%, 64.8%]** | **0.580 [0.540, 0.619]** | **0.626 [0.582, 0.669]** | Multimodal Imaging Fusion |
| **ROI Synchrony (190 ROIs)** | **XGBoost** | **60.4% [56.6%, 64.2%]** | **0.593 [0.554, 0.631]** | **0.626 [0.581, 0.670]** | Kendall's $\tau$ ReHo Metric |
| **FC PCA (488 PCs)** | **XGBoost** | **57.1% [53.4%, 61.0%]** | **0.555 [0.517, 0.594]** | **0.569 [0.525, 0.612]** | 95% Variance PCA |
| *Dummy Baseline* | *Stratified* | *55.8% [51.9%, 59.7%]* | *0.358 [0.320, 0.395]* | *0.500 [0.500, 0.500]* | Random Prior Baseline |

### Full 5 Models × 6 Feature Sets Accuracy Matrix (%)

| Model | FC_PCA | ROI Synchrony | fALFF | Combined | FC (Full) | FC + Pheno |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy Baseline** | 54.9 | 54.9 | 54.8 | 54.9 | 54.9 | 55.8 |
| **Decision Tree** | 50.0 | 56.5 | 55.7 | 51.3 | 52.7 | 70.8 |
| **Deep Neural Net (DNN)** | 52.7 | 56.5 | 54.0 | 54.2 | 61.8 | 60.7 |
| **SVM (RBF Kernel)** | 53.0 | 57.2 | 59.4 | 60.6 | 62.3 | 63.0 |
| **Random Forest** | 53.0 | 60.2 | 60.3 | 59.0 | 60.3 | 63.5 |
| **XGBoost** | **57.1** | **60.4** | **61.7** | **61.0** | 61.8 | **76.6** |

---

## Dataset & Feature Extraction

### ADHD-200 Sample Composition
- **Total Sample Size**: $N = 620$ subjects across 7 international collection sites.
- **Diagnostic Distribution**: 340 Typically Developing Controls (54.8%), 280 ADHD (45.2%).
- **Site Breakdown**:
  - New York University (NYU): 216 subjects
  - Peking University (Peking 1, 2, 3): 194 subjects
  - Kennedy Krieger Institute (KKI): 83 subjects
  - Oregon Health & Science University (OHSU): 79 subjects
  - NeuroIMAGE: 48 subjects
- **Preprocessing Pipeline**: Preprocessed via the **Athena Pipeline** (slice timing correction, motion correction, 6mm FWHM spatial smoothing, 0.009–0.08 Hz bandpass filtering, CC200 ROI time series extraction across 190 regions).

### Feature Sets
1. **Functional Connectivity (FC)**: Pearson correlation coefficient matrix ($190 \times 190$) flattened upper triangle = **17,955 features**.
2. **FC PCA**: Principal Component Analysis reducing FC to **488 components** (retaining 95% variance). PCA is fitted strictly within cross-validation training folds.
3. **fALFF (Fractional Amplitude of Low-Frequency Fluctuations)**: Ratio of low-frequency BOLD power (0.01–0.08 Hz) to total signal power per ROI = **190 features**.
4. **ROI Synchrony Approximation**: Regional Homogeneity (ReHo) approximation computed via average pairwise Kendall's $\tau$ correlation per ROI = **190 features**.
5. **Combined Imaging**: Concatenation of FC PCA (488) + ROI Sync (190) + fALFF (190) = **868 features**.
6. **FC + Phenotypic**: Full FC (17,955) + 6 clinical metadata variables = **17,961 features**.
   - *Phenotypic Variables Included*: Verbal IQ, Performance IQ, Full4 IQ, Age, Sex, Medication Status.
   - *Leaky Variables Excluded*: ADHD Index, Inattentive Subscore, Hyperactive/Impulsive Subscore (circular diagnostic subscale scores).

---

## Model Architectures & Methodological Safeguards

### Evaluated Algorithms
- **Support Vector Classifier (SVM)**: Radial Basis Function (RBF) kernel with inner grid search over regularization $C \in [0.1, 1, 10, 100]$ and gamma $\gamma \in [\text{'scale'}, \text{'auto'}, 0.001, 0.01]$.
- **Decision Tree**: Single decision tree using Gini impurity split criterion.
- **Random Forest**: Ensemble of 100 decision trees with bootstrap sampling and feature sub-selection.
- **XGBoost**: Gradient-boosted decision tree ensemble with max depth tuning and tree regularization.
- **Deep Neural Network (DNN)**: Multi-Layer Perceptron (MLP) built with Keras/TensorFlow. Architecture: Dense(256) $\rightarrow$ Batch Normalization $\rightarrow$ ReLU $\rightarrow$ Dropout(0.4) $\rightarrow$ Dense(128) $\rightarrow$ Batch Normalization $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ Dense(64) $\rightarrow$ ReLU $\rightarrow$ Dense(1, Sigmoid). Trained using Adam optimizer, binary cross-entropy, class weight balancing, and early stopping on validation loss.

### Data Leakage Safeguards
1. **Strict Cross-Validation**: Stratified 10-fold CV evaluates out-of-fold predictions.
2. **Preprocessing Isolation**: Feature standardization (`StandardScaler`) and PCA dimensionality reduction are fitted exclusively on training folds and transformed onto test folds.
3. **No Target Leakage**: Direct clinical rating subscores (ADHD Index, Inattentive, Hyperactive) were deleted from model inputs.

---

## Repository Structure

```
.
├── paper/                                                     # Publication manuscript & TeX source
│   ├── paper.tex                                              # Corrected IEEE format LaTeX source
│   ├── Comparing Machine Learning Algorithms...pdf            # Compiled 5-page IEEE manuscript PDF
│   ├── README.md                                              # LaTeX compilation guide
│   └── figures/                                               # Publication-ready figures
│       ├── FC_Pheno_model_comparison.png                      # Model comparison bar chart with CIs
│       ├── FC_Pheno_roc_curves.png                            # ROC curves for all models
│       ├── accuracy_heatmap.png                               # Models x Datasets accuracy heatmap
│       ├── FC_Pheno_confusion_matrices.png                    # Grid of confusion matrices
│       ├── falff_subject_1186237.png                          # Preprocessing: fALFF map
│       ├── fc_heatmap_subject_1133221_final.png               # Preprocessing: FC matrix
│       ├── reho_sync_subject_1133221_final.png                # Preprocessing: Regional synchrony
│       └── pca_scree_plot_95percent_threshold.png             # Preprocessing: PCA scree plot
├── src/                                                       # Modular Python pipeline
│   ├── config.py                                              # Global configuration parameters
│   ├── data_loader.py                                         # Dataset loading & 6 feature configs
│   ├── train.py                                               # Training orchestrator (30 experiments)
│   ├── evaluate.py                                            # Bootstrap CIs & McNemar's tests
│   ├── visualize.py                                           # Publication chart generation
│   ├── inspect_and_prepare_data.py                            # Data inspection utility
│   └── models/                                                # Model implementations
│       ├── base.py                                            # Base model class (CV + OOF evaluator)
│       ├── svm.py                                             # SVM (RBF) classifier
│       ├── decision_tree.py                                   # Decision Tree classifier
│       ├── random_forest.py                                   # Random Forest ensemble
│       ├── xgboost_model.py                                   # XGBoost classifier
│       ├── dnn_tf.py                                          # Keras/TensorFlow DNN (MLP)
│       └── dummy_baseline.py                                  # Stratified baseline
├── Feature Extraction jupyter notebooks/                      # Feature extraction notebooks
├── Machine Learning Models/                                  # Initial exploratory notebooks
├── data/                                                      # Data directory (gitignored)
│   ├── raw/                                                   # Raw Athena CSV files
│   ├── processed/                                             # Preprocessed arrays (X.npy, y.npy)
│   ├── README.md                                              # Dataset documentation
│   └── SOURCE.txt                                             # ADHD-200 NITRC download link
├── results/                                                   # Experiment outputs
│   ├── {Model}/{Dataset}/                                     # Per-fold metrics & OOF predictions
│   ├── bootstrap_ci.csv                                       # Bootstrap 95% confidence intervals
│   ├── mcnemar_tests.csv                                      # McNemar statistical test results
│   ├── evaluation_results.json                                # Combined metrics JSON
│   └── charts/                                                # Pipeline chart outputs
├── run_0_setup.ps1                                            # Step 0: Setup environment
├── run_1_train_fast.ps1                                       # Step 1: Train Baseline + DecisionTree
├── run_2_train_svm.ps1                                        # Step 2: Train SVM
├── run_3_train_rf_xgb.ps1                                     # Step 3: Train Random Forest + XGBoost
├── run_4_train_dnn.ps1                                        # Step 4: Train DNN
├── run_5_evaluate.ps1                                         # Step 5: Evaluate Bootstrap + McNemar
├── run_6_visualize.ps1                                        # Step 6: Generate figures
├── run_all.ps1                                                # Master orchestration script
├── requirements.txt                                           # Python dependencies
└── README.md                                                  # Main project documentation
```

---

## Quick Start & Reproduction

### Prerequisites
- Python 3.9+
- PowerShell (on Windows) or Bash (on Linux/macOS)

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Prajwalparajuli/Comparing-Machine-Learning-Algorithms-for-ADHD-Diagnosis.git
cd Comparing-Machine-Learning-Algorithms-for-ADHD-Diagnosis
pip install -r requirements.txt
```

### 2. Download Data
Download the ADHD-200 preprocessed Athena pipeline CSVs from [NITRC NeuroBureau](https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline) and place the raw CSV files under `data/raw/`.

### 3. Run Pipeline via Python Modules
To train all 30 model-dataset combinations, evaluate statistics, and generate publication charts:

```bash
# Train all 5 models across all 6 feature sets
python -m src.train

# Calculate 2000-iteration bootstrap 95% CIs and McNemar's tests
python -m src.evaluate

# Generate publication figures in results/charts/ and paper/figures/
python -m src.visualize
```

### 4. Run Pipeline via PowerShell Scripts
Alternatively, execute step-by-step PowerShell scripts:

```powershell
.\run_0_setup.ps1           # Environment setup
.\run_1_train_fast.ps1      # Train Dummy Baseline & Decision Tree
.\run_2_train_svm.ps1       # Train SVM (RBF)
.\run_3_train_rf_xgb.ps1    # Train Random Forest & XGBoost
.\run_4_train_dnn.ps1       # Train Keras Deep Neural Network
.\run_5_evaluate.ps1        # Compute Bootstrap CIs & McNemar's tests
.\run_6_visualize.ps1       # Generate publication charts
```

Or run everything end-to-end with a single command:
```powershell
.\run_all.ps1
```

---

## Compiling the LaTeX Paper

To compile the manuscript PDF locally:

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex    # Run twice for cross-references
```

Requires a working LaTeX distribution with `pdflatex` (e.g., MiKTeX or TeX Live) and the `IEEEtran` class package.

---

## License & Citation

This project is released under the [MIT License](LICENSE).

If you find this codebase or manuscript useful in your research, please cite:

```bibtex
@inproceedings{parajuli2026adhd,
  title={Comparing Machine Learning Algorithms for ADHD Diagnosis Using Multimodal rs-fMRI and Clinical Features from the ADHD-200 Dataset},
  author={Parajuli, Prajwal},
  booktitle={Department of Mathematics and Statistics, University of Houston-Downtown},
  year={2026}
}
```
