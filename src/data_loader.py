"""
Data loading utilities.

Supports loading 6 separate preprocessed datasets:
  1. FC           — raw Functional Connectivity (17,955 features)
  2. FC_Pheno     — FC + Phenotypic (age, sex, IQ, site, etc.)
  3. FC_PCA       — PCA-reduced FC (488 components, pre-computed)
  4. ROI_Sync     — ROI Synchrony (190 features)
  5. fALFF        — fractional ALFF (190 features)
  6. Combined     — FC_PCA + ROI_Sync + fALFF concatenated (868 features)

Each dataset returns (X, y) with X as the feature matrix and y as
the binary label vector (0=Control, 1=ADHD).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import RAW_DIR, DATA_DIR


# ─── Dataset Registry ──────────────────────────────────────────
DATASET_NAMES = [
    "FC_PCA",
    "ROI_Sync",
    "fALFF",
    "Combined",
    "FC",
    "FC_Pheno",
]

DATASET_DESCRIPTIONS = {
    "FC":        "Functional Connectivity (CC200 correlations)",
    "FC_Pheno":  "Functional Connectivity + Phenotypic features",
    "FC_PCA":    "PCA-reduced Functional Connectivity (488 components)",
    "ROI_Sync":  "ROI Synchrony (regional homogeneity proxy)",
    "fALFF":     "Fractional Amplitude of Low-Frequency Fluctuations",
    "Combined":  "FC_PCA + ROI_Sync + fALFF concatenated",
}


def _load_csv(path, name):
    """Load a CSV and normalize its index to ScanDir ID (int)."""
    df = pd.read_csv(path, index_col=0)
    if "ScanDir ID" in df.columns:
        df = df.set_index("ScanDir ID")
    df.index.name = "ScanDir ID"
    df.index = df.index.astype(int)
    # Binarize DX if present
    if "DX" in df.columns:
        df["DX"] = df["DX"].apply(lambda x: 1 if x > 0 else 0)
    return df


def _extract_Xy(df):
    """Split a DataFrame into feature matrix X and label vector y."""
    feature_cols = [c for c in df.columns if c != "DX"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["DX"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def load_dataset(name):
    """
    Load a named dataset.

    Parameters
    ----------
    name : str
        One of DATASET_NAMES.

    Returns
    -------
    X : np.ndarray (n_samples, n_features)
    y : np.ndarray (n_samples,)
    desc : str
    """
    assert name in DATASET_NAMES, f"Unknown dataset: {name}. Choose from {DATASET_NAMES}"

    if name == "FC":
        df = _load_csv(RAW_DIR / "Merged_FC.csv", "FC")
        X, y = _extract_Xy(df)

    elif name == "FC_Pheno":
        df = _load_csv(RAW_DIR / "FC_Merged_Pheno.csv", "FC_Pheno")
        # CRITICAL: Drop diagnostic scores — they ARE the diagnosis
        # Including them would give ~95% accuracy but that's circular.
        leak_cols = ["ADHD Index", "Inattentive", "Hyper/Impulsive"]
        dropped = [c for c in leak_cols if c in df.columns]
        if dropped:
            df = df.drop(columns=dropped)
            print(f"    [WARNING] Dropped leaky columns: {dropped}", flush=True)
        X, y = _extract_Xy(df)

    elif name == "FC_PCA":
        df = _load_csv(RAW_DIR / "PCA_feature.csv", "FC_PCA")
        X, y = _extract_Xy(df)

    elif name == "ROI_Sync":
        df = _load_csv(RAW_DIR / "All_ReHo.csv", "ROI_Sync")
        X, y = _extract_Xy(df)

    elif name == "fALFF":
        df = _load_csv(RAW_DIR / "All_falff.csv", "fALFF")
        X, y = _extract_Xy(df)

    elif name == "Combined":
        pca_df = _load_csv(RAW_DIR / "PCA_feature.csv", "FC_PCA")
        reho_df = _load_csv(RAW_DIR / "All_ReHo.csv", "ROI_Sync")
        falff_df = _load_csv(RAW_DIR / "All_falff.csv", "fALFF")

        common = sorted(
            set(pca_df.index) & set(reho_df.index) & set(falff_df.index)
        )
        pca_df = pca_df.loc[common]
        reho_df = reho_df.loc[common]
        falff_df = falff_df.loc[common]

        y = pca_df["DX"].values.astype(np.int32)

        pca_cols = [c for c in pca_df.columns if c != "DX"]
        reho_cols = [c for c in reho_df.columns if c != "DX"]
        falff_cols = [c for c in falff_df.columns if c != "DX"]

        X = np.hstack([
            pca_df[pca_cols].values,
            reho_df[reho_cols].values,
            falff_df[falff_cols].values,
        ]).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    desc = DATASET_DESCRIPTIONS[name]
    print(f"  [{name}] {desc}")
    print(f"    X: {X.shape}  y: {y.shape}  "
          f"Control={np.sum(y==0)} ADHD={np.sum(y==1)}")

    return X, y, desc


def load_all_datasets():
    """Load all datasets and return as a dict."""
    datasets = {}
    print(f"\n{'='*60}")
    print(f"  Loading all {len(DATASET_NAMES)} datasets")
    print(f"{'='*60}")
    for name in DATASET_NAMES:
        try:
            X, y, desc = load_dataset(name)
            datasets[name] = {"X": X, "y": y, "desc": desc}
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    return datasets


# Backward compatibility
def load_data():
    """Legacy loader — returns Combined dataset."""
    X, y, desc = load_dataset("Combined")
    return X, y, None
