# Paper and Publication Assets

This directory contains manuscript and publication-related materials.

## Files
- **paper.tex** — Corrected LaTeX source (IEEE conference format)
- **Comparing Machine Learning Algorithms for ADHD Diagnosis Using Multimodal rs-fMRI and Clinical Features from the ADHD-200 Dataset.pdf** — Compiled manuscript
- **figures/** — Publication figures (corrected charts from pipeline + preprocessing visualizations)

## Compiling
```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex   # Run twice for references
```

## Figures
- `accuracy_heatmap.png` — Models × Datasets accuracy heatmap
- `FC_Pheno_model_comparison.png` — Bar chart with bootstrap CIs
- `FC_Pheno_roc_curves.png` — ROC curves for all models
- `FC_Pheno_confusion_matrices.png` — Confusion matrix grid
- `falff_subject_1186237.png` — fALFF feature visualization
- `fc_heatmap_subject_1133221_final.png` — FC matrix heatmap
- `reho_sync_subject_1133221_final.png` — ReHo synchrony visualization
- `pca_scree_plot_95percent_threshold.png` — PCA variance explained