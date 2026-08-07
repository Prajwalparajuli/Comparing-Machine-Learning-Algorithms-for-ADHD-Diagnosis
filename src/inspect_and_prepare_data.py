"""
Inspect the raw processed CSV files and prepare X.npy / y.npy / metadata.csv
for the ML pipeline.

This script:
1. Loads each CSV from data/raw/
2. Reports shape, dtype, class distribution, value ranges
3. Generates data/processed/X.npy, y.npy, metadata.csv
4. Validates the output
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def inspect_csv(name, path):
    """Load and inspect a CSV file."""
    print(f"\n{'='*60}")
    print(f"  {name}: {path.name}")
    print(f"{'='*60}")
    df = pd.read_csv(path, index_col=0)
    print(f"  Shape:       {df.shape}")
    print(f"  Index name:  {df.index.name}")
    print(f"  Columns:     {list(df.columns[:5])} ... {list(df.columns[-5:])}")
    print(f"  Index range: [{df.index.min()}, {df.index.max()}]")
    print(f"  dtypes:      {df.dtypes.value_counts().to_dict()}")
    print(f"  NaN total:   {df.isna().sum().sum()}")
    
    if "DX" in df.columns:
        dx = df["DX"]
        print(f"\n  DX distribution:")
        for val, cnt in dx.value_counts().sort_index().items():
            pct = cnt / len(dx) * 100
            print(f"    DX={val}: {cnt} ({pct:.1f}%)")
        print(f"  DX unique:   {sorted(dx.unique())}")
    
    # Value ranges for numeric columns (excluding DX)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != "DX"]
    if feature_cols:
        feature_data = df[feature_cols]
        print(f"\n  Feature stats (excl DX):")
        print(f"    Range: [{feature_data.min().min():.6f}, {feature_data.max().max():.6f}]")
        print(f"    Mean:  {feature_data.values.mean():.6f}")
        print(f"    Std:   {feature_data.values.std():.6f}")
    
    return df


def normalize_index(df, name):
    """Ensure the DataFrame uses ScanDir ID as its index."""
    if "ScanDir ID" in df.columns:
        print(f"  [{name}] Setting 'ScanDir ID' column as index")
        df = df.set_index("ScanDir ID")
    df.index.name = "ScanDir ID"
    df.index = df.index.astype(int)
    return df


def prepare_processed_data(fc_pca_df, reho_df, falff_df, fc_pheno_df):
    """
    Prepare X.npy and y.npy from the processed CSVs.
    
    Strategy: Use PCA-reduced FC features (488 PCs) + ReHo (190) + fALFF (190)
    = 868 features total.
    
    We DO NOT include phenotypic features in X because:
    - ADHD Index, Inattentive, Hyper/Impulsive scores are effectively labels (circular)
    - IQ and Med Status are clinical covariates we may use separately
    """
    print(f"\n{'='*60}")
    print(f"  Preparing Processed Data")
    print(f"{'='*60}")
    
    # 0. Normalize all DataFrames to use ScanDir ID as index
    fc_pca_df = normalize_index(fc_pca_df.copy(), "FC PCA")
    reho_df = normalize_index(reho_df.copy(), "ReHo")
    falff_df = normalize_index(falff_df.copy(), "fALFF")
    
    # 1. Get common subjects across all modalities
    fc_subjects = set(fc_pca_df.index)
    reho_subjects = set(reho_df.index)
    falff_subjects = set(falff_df.index)
    
    common = fc_subjects & reho_subjects & falff_subjects
    print(f"\n  FC PCA subjects:  {len(fc_subjects)}")
    print(f"  ReHo subjects:    {len(reho_subjects)}")
    print(f"  fALFF subjects:   {len(falff_subjects)}")
    print(f"  Common subjects:  {len(common)}")
    
    if len(common) == 0:
        # Debug: show sample indices from each
        print(f"\n  DEBUG — FC PCA index sample: {sorted(list(fc_subjects))[:5]}")
        print(f"  DEBUG — ReHo index sample:   {sorted(list(reho_subjects))[:5]}")
        print(f"  DEBUG — fALFF index sample:   {sorted(list(falff_subjects))[:5]}")
        raise ValueError("No common subjects found! Check index alignment.")
    
    # Sort for reproducibility
    common_sorted = sorted(common)
    
    # 2. Extract labels (DX) from FC PCA (it has the DX column)
    y_raw = fc_pca_df.loc[common_sorted, "DX"].values
    
    # Binarize: 0=Control, 1/2/3 → 1=ADHD
    y = np.where(y_raw == 0, 0, 1).astype(np.int32)
    
    print(f"\n  Label binarization:")
    print(f"    Raw DX values: {sorted(np.unique(y_raw))}")
    for val in sorted(np.unique(y_raw)):
        cnt = (y_raw == val).sum()
        print(f"      DX={val}: {cnt}")
    print(f"    After binarization:")
    for val in [0, 1]:
        cnt = (y == val).sum()
        pct = cnt / len(y) * 100
        label = "Control" if val == 0 else "ADHD"
        print(f"      {label} (y={val}): {cnt} ({pct:.1f}%)")
    
    # 3. Extract features
    # FC PCA features (drop DX column)
    fc_pca_cols = [c for c in fc_pca_df.columns if c != "DX"]
    X_fc_pca = fc_pca_df.loc[common_sorted, fc_pca_cols].values
    
    # ReHo features (drop DX if present)
    reho_cols = [c for c in reho_df.columns if c != "DX"]
    X_reho = reho_df.loc[common_sorted, reho_cols].values
    
    # fALFF features (drop DX if present)
    falff_cols = [c for c in falff_df.columns if c != "DX"]
    X_falff = falff_df.loc[common_sorted, falff_cols].values
    
    print(f"\n  Feature dimensions:")
    print(f"    FC PCA:  {X_fc_pca.shape}")
    print(f"    ReHo:    {X_reho.shape}")
    print(f"    fALFF:   {X_falff.shape}")
    
    # 4. Concatenate features
    X = np.hstack([X_fc_pca, X_reho, X_falff]).astype(np.float64)
    print(f"    Combined: {X.shape}")
    
    # 5. Check for NaN/Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"\n  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    
    if nan_count > 0:
        print("  WARNING: NaN values found! Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)
    if inf_count > 0:
        print("  WARNING: Inf values found! Replacing with 0.")
        X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
    
    # 6. Create metadata with site info
    # Load per-site CSVs to build a subject → site mapping
    site_map = {}
    site_csv_files = {
        "KKI": RAW_DIR.parent.parent / "data" / "raw" / "site_maps" / "dummy",  # placeholder
    }
    # Build site mapping from per-site FC CSVs if available
    src_datasets = Path(r"C:\Users\prajw\Desktop\College Files\Undergrad Research\Datasets")
    site_names = ["KKI", "NeuroIMAGE", "NYU", "OHSU", "Peking_1", "Peking_2", "Peking_3"]
    for site_name in site_names:
        site_csv = src_datasets / "Preprocessed FC" / f"{site_name}_FC.csv"
        if site_csv.exists():
            site_df = pd.read_csv(site_csv)
            if "ScanDir ID" in site_df.columns:
                for sid in site_df["ScanDir ID"].values:
                    site_map[int(sid)] = site_name
    
    sites = [site_map.get(sid, "Unknown") for sid in common_sorted]
    
    metadata = pd.DataFrame({
        "subject_id": common_sorted,
        "site": sites,
        "DX_original": y_raw,
        "DX_binary": y
    })
    
    print(f"\n  Metadata created with {len(metadata)} subjects")
    print(f"  Site distribution:")
    for site, cnt in metadata["site"].value_counts().sort_index().items():
        print(f"    {site}: {cnt}")
    
    # 7. Save
    np.save(PROC_DIR / "X.npy", X)
    np.save(PROC_DIR / "y.npy", y)
    metadata.to_csv(PROC_DIR / "metadata.csv", index=False)
    
    print(f"\n  Saved:")
    print(f"    {PROC_DIR / 'X.npy'} — shape {X.shape}, dtype {X.dtype}")
    print(f"    {PROC_DIR / 'y.npy'} — shape {y.shape}, dtype {y.dtype}")
    print(f"    {PROC_DIR / 'metadata.csv'} — {len(metadata)} rows")
    
    # 8. Summary JSON
    unique, counts = np.unique(y, return_counts=True)
    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_fc_pca_features": int(X_fc_pca.shape[1]),
        "n_reho_features": int(X_reho.shape[1]),
        "n_falff_features": int(X_falff.shape[1]),
        "n_classes": int(len(unique)),
        "class_distribution": {
            "Control": int(counts[0]),
            "ADHD": int(counts[1])
        },
        "class_ratio": f"{counts[0]}:{counts[1]}",
        "feature_range": [float(X.min()), float(X.max())],
        "feature_mean": float(X.mean()),
        "feature_std": float(X.std()),
    }
    
    with open(PROC_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    {PROC_DIR / 'data_summary.json'}")
    
    return X, y, metadata


def main():
    print("=" * 60)
    print("  ADHD-200 Data Inspection & Preparation")
    print("=" * 60)
    
    # 1. Inspect each CSV
    merged_fc = inspect_csv("Merged FC", RAW_DIR / "Merged_FC.csv")
    pca_fc = inspect_csv("PCA Features", RAW_DIR / "PCA_feature.csv")
    fc_pheno = inspect_csv("FC + Phenotypic", RAW_DIR / "FC_Merged_Pheno.csv")
    reho = inspect_csv("ReHo", RAW_DIR / "All_ReHo.csv")
    falff = inspect_csv("fALFF", RAW_DIR / "All_falff.csv")
    
    # 2. Prepare processed data
    X, y, metadata = prepare_processed_data(pca_fc, reho, falff, fc_pheno)
    
    # 3. Final validation
    print(f"\n{'='*60}")
    print(f"  FINAL VALIDATION")
    print(f"{'='*60}")
    print(f"  X shape:     {X.shape}")
    print(f"  y shape:     {y.shape}")
    print(f"  X[0,:5]:     {X[0,:5]}")
    print(f"  y[:10]:      {y[:10]}")
    unique, counts = np.unique(y, return_counts=True)
    print(f"  Control:     {counts[0]} ({counts[0]/len(y)*100:.1f}%)")
    print(f"  ADHD:        {counts[1]} ({counts[1]/len(y)*100:.1f}%)")
    print(f"  Balance:     {counts.min()/counts.max():.3f}")
    print(f"\n  ✓ All files saved to {PROC_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
