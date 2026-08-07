"""
Publication-quality visualization — multi-dataset support.

Reads from results/{MODEL}/{DATASET}/ and bootstrap_ci.csv.

Generates:
  1. Per-dataset bar charts (Acc/F1/AUC with CIs)
  2. Heatmap matrix: Models × Datasets (accuracy)
  3. Per-dataset confusion matrix grids
  4. Per-dataset ROC curve overlays
  5. Per-dataset fold boxplots
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

from src.config import (
    RESULTS_DIR, CHARTS_DIR, MODEL_NAMES,
    DPI, PALETTE, FONT_FAMILY,
)
from src.data_loader import DATASET_NAMES


def setup_style():
    """Set publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def discover_data():
    """Load fold metrics and OOF predictions from all model/dataset dirs."""
    fold_frames = []
    oof_data = {}  # {dataset: {model: DataFrame}}

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name

            fold_path = ds_dir / "fold_metrics.csv"
            if fold_path.exists():
                fold_frames.append(pd.read_csv(fold_path))

            oof_path = ds_dir / "oof_predictions.csv"
            if oof_path.exists():
                if ds_name not in oof_data:
                    oof_data[ds_name] = {}
                oof_data[ds_name][model_name] = pd.read_csv(oof_path)

    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    return fold_df, oof_data


def plot_heatmap_matrix(ci_df):
    """Heatmap: Models × Datasets accuracy matrix."""
    pivot = ci_df.pivot_table(index="model", columns="dataset",
                              values="acc_mean", aggfunc="first")

    # Order
    model_order = [m for m in MODEL_NAMES if m in pivot.index]
    ds_order = [d for d in DATASET_NAMES if d in pivot.columns]
    pivot = pivot.reindex(index=model_order, columns=ds_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0.4, vmax=1.0,
                annot_kws={"size": 12, "fontweight": "bold"})
    ax.set_title("Accuracy: Models × Datasets", fontsize=16, fontweight="bold")
    ax.set_xlabel("Preprocessing / Feature Set")
    ax.set_ylabel("Model")

    plt.tight_layout()
    path = CHARTS_DIR / "accuracy_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_dataset_comparison(ci_df, ds_name, fold_df):
    """Bar chart for a single dataset comparing all models."""
    df = ci_df[ci_df["dataset"] == ds_name]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Model Comparison — {ds_name}", fontsize=16, fontweight="bold")

    for ax, (key, label) in zip(axes, [("acc", "Accuracy"), ("f1", "F1-Macro"), ("auc", "ROC-AUC")]):
        models = df["model"].values
        means = df[f"{key}_mean"].values
        los = df[f"{key}_lo"].values
        his = df[f"{key}_hi"].values
        errors = np.array([means - los, his - means])
        colors = [PALETTE.get(m, "#888888") for m in models]

        ax.barh(range(len(models)), means, xerr=errors,
                color=colors, edgecolor="white", linewidth=0.5,
                error_kw={"capsize": 3, "capthick": 1, "elinewidth": 1},
                height=0.7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

        for i, m in enumerate(means):
            if not np.isnan(m):
                ax.text(m + 0.01, i, f"{m:.3f}", va="center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ds_chart_dir = CHARTS_DIR / ds_name
    ds_chart_dir.mkdir(parents=True, exist_ok=True)
    path = ds_chart_dir / "model_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices(oof_data, ds_name):
    """Grid of confusion matrices for one dataset."""
    models = oof_data.get(ds_name, {})
    if not models:
        return

    available = [m for m in MODEL_NAMES if m in models]
    n = len(available)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(f"Confusion Matrices — {ds_name}", fontsize=16, fontweight="bold")

    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(available):
        df = models[name]
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Control", "ADHD"],
                    yticklabels=["Control", "ADHD"],
                    ax=axes[i], cbar=False,
                    annot_kws={"size": 14, "fontweight": "bold"})
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ds_chart_dir = CHARTS_DIR / ds_name
    ds_chart_dir.mkdir(parents=True, exist_ok=True)
    path = ds_chart_dir / "confusion_matrices.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(oof_data, ds_name):
    """ROC curves overlay for one dataset."""
    models = oof_data.get(ds_name, {})
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"ROC Curves — {ds_name}", fontsize=16, fontweight="bold")

    for name in MODEL_NAMES:
        if name not in models:
            continue
        df = models[name]
        if "y_proba" not in df.columns:
            continue
        y_true = df["y_true"].values
        y_proba = df["y_proba"].values
        if np.all(np.isnan(y_proba)):
            continue

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc_val = auc(fpr, tpr)
        color = PALETTE.get(name, "#888888")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC={roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    ds_chart_dir = CHARTS_DIR / ds_name
    ds_chart_dir.mkdir(parents=True, exist_ok=True)
    path = ds_chart_dir / "roc_curves.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  ADHD ML Pipeline — Visualization")
    print("=" * 60)

    setup_style()

    fold_df, oof_data = discover_data()

    ci_path = RESULTS_DIR / "bootstrap_ci.csv"
    if not ci_path.exists():
        print("  [ERROR] No bootstrap_ci.csv found. Run evaluate.py first.")
        return
    ci_df = pd.read_csv(ci_path)

    datasets_found = sorted(oof_data.keys())
    print(f"  Datasets: {datasets_found}")

    # 1. Overall heatmap
    plot_heatmap_matrix(ci_df)

    # 2. Per-dataset charts
    for ds_name in datasets_found:
        print(f"\n  --- {ds_name} ---")
        plot_dataset_comparison(ci_df, ds_name, fold_df)
        plot_confusion_matrices(oof_data, ds_name)
        plot_roc_curves(oof_data, ds_name)

    print(f"\n  All charts saved to {CHARTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
