"""
Base training harness shared by all model scripts.

Provides:
  1. run_experiment()      — single model × single dataset CV
  2. run_across_datasets() — single model × ALL datasets
  3. Results save to results/{MODEL}/{DATASET}/
"""
import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from src.config import N_FOLDS, RANDOM_STATE, RESULTS_DIR
from src.data_loader import DATASET_NAMES, load_dataset

warnings.filterwarnings("ignore")


def get_cv():
    """Standard stratified k-fold used by ALL models."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def run_experiment(model_name, estimator, X, y, dataset_name="Combined",
                   is_search=False):
    """
    Run stratified k-fold CV for a single model on a single dataset.

    Parameters
    ----------
    model_name : str
    estimator : sklearn estimator / Pipeline / SearchCV
    X : np.ndarray
    y : np.ndarray
    dataset_name : str
    is_search : bool — if True, log best_params_ from SearchCV.

    Returns
    -------
    dict — results summary.
    """
    skf = get_cv()
    fold_metrics = []
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), np.nan, dtype=float)
    best_params_per_fold = []

    print(f"\n{'='*60}")
    print(f"  {model_name} × {dataset_name} — {N_FOLDS}-Fold CV")
    print(f"{'='*60}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        t0 = time.time()
        estimator.fit(X_train, y_train)
        train_time = time.time() - t0

        if is_search and hasattr(estimator, "best_params_"):
            best_params_per_fold.append(estimator.best_params_)

        y_pred = estimator.predict(X_val)
        oof_preds[val_idx] = y_pred

        try:
            y_prob = estimator.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = y_prob
            auc = roc_auc_score(y_val, y_prob)
        except Exception:
            auc = np.nan

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        fold_metrics.append({
            "model": model_name,
            "dataset": dataset_name,
            "fold": fold_idx + 1,
            "accuracy": acc,
            "f1_macro": f1,
            "roc_auc": auc,
            "train_time_s": round(train_time, 2),
        })

        print(f"  Fold {fold_idx+1:2d} | Acc={acc:.4f}  F1={f1:.4f}  "
              f"AUC={auc:.4f}  ({train_time:.1f}s)")

    # Summary stats
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    aucs = [m["roc_auc"] for m in fold_metrics if not np.isnan(m["roc_auc"])]

    print(f"\n  >>> {model_name} × {dataset_name}:")
    print(f"      Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"      F1-macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    if aucs:
        print(f"      ROC-AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Save to results/{MODEL}/{DATASET}/
    save_dir = RESULTS_DIR / model_name / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(fold_metrics).to_csv(save_dir / "fold_metrics.csv", index=False)

    oof_df = pd.DataFrame({
        "y_true": y,
        "y_pred": oof_preds,
        "y_proba": oof_proba,
    })
    oof_df.to_csv(save_dir / "oof_predictions.csv", index=False)

    report = classification_report(y, oof_preds, output_dict=True,
                                   target_names=["Control", "ADHD"],
                                   zero_division=0)
    report["confusion_matrix"] = confusion_matrix(y, oof_preds).tolist()
    with open(save_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if best_params_per_fold:
        with open(save_dir / "best_params.json", "w") as f:
            json.dump(best_params_per_fold, f, indent=2, default=str)
        print(f"      Best params (fold 1): {best_params_per_fold[0]}")

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "auc_mean": np.mean(aucs) if aucs else np.nan,
        "auc_std": np.std(aucs) if aucs else np.nan,
    }

    print(f"  Results → {save_dir}/")
    return summary


def run_across_datasets(model_name, build_fn, datasets=None, is_search=False):
    """
    Run a model across all (or specified) datasets.

    Parameters
    ----------
    model_name : str
    build_fn : callable() -> estimator  (returns a fresh estimator each call)
    datasets : list[str] or None  (None = all datasets)
    is_search : bool

    Returns
    -------
    list[dict]
    """
    if datasets is None:
        datasets = DATASET_NAMES

    summaries = []
    for ds_name in datasets:
        try:
            X, y, desc = load_dataset(ds_name)
            model = build_fn()
            summary = run_experiment(
                model_name, model, X, y,
                dataset_name=ds_name,
                is_search=is_search,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\n  [ERROR] {model_name} × {ds_name}: {e}")

    return summaries
