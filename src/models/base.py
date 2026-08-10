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
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
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
    Run 10-fold CV for a single model on a single dataset.
    If estimator is a SearchCV object (or is_search=True), hyperparameter search
    is performed ONCE on the full dataset, and the best estimator is evaluated across 10 folds.

    Parameters
    ----------
    model_name : str
    estimator : sklearn estimator / Pipeline / SearchCV
    X : np.ndarray
    y : np.ndarray
    dataset_name : str
    is_search : bool

    Returns
    -------
    dict — results summary.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  {model_name} × {dataset_name} — Evaluation Harness", flush=True)
    print(f"{'='*60}", flush=True)

    best_params = None
    is_search_obj = isinstance(estimator, (RandomizedSearchCV, GridSearchCV))

    if is_search_obj or (is_search and hasattr(estimator, "fit")):
        if is_search_obj:
            print(f"  [Step 1/2] Hyperparameter Tuning (Search ONCE on dataset)...", flush=True)
            t0_search = time.time()
            estimator.fit(X, y)
            search_time = time.time() - t0_search
            best_params = estimator.best_params_
            eval_estimator = estimator.best_estimator_
            print(f"            Search complete in {search_time:.1f}s", flush=True)
            print(f"            Best Params: {best_params}", flush=True)
        else:
            eval_estimator = estimator
    else:
        eval_estimator = estimator

    skf = get_cv()
    fold_metrics = []
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_proba = np.full(len(y), np.nan, dtype=float)

    print(f"  [Step 2/2] 10-Fold Cross-Validation Evaluation...", flush=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_model = clone(eval_estimator)
        t0 = time.time()
        fold_model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = fold_model.predict(X_val)
        oof_preds[val_idx] = y_pred

        try:
            y_prob = fold_model.predict_proba(X_val)[:, 1]
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

        print(f"    Fold {fold_idx+1:2d} | Acc={acc:.4f}  F1={f1:.4f}  "
              f"AUC={auc:.4f}  ({train_time:.1f}s)", flush=True)

    # Summary stats
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    aucs = [m["roc_auc"] for m in fold_metrics if not np.isnan(m["roc_auc"])]

    print(f"\n  >>> {model_name} x {dataset_name}:", flush=True)
    print(f"      Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)
    print(f"      F1-macro: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", flush=True)
    if aucs:
        print(f"      ROC-AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}", flush=True)

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

    if best_params:
        with open(save_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2, default=str)
        print(f"      Best params saved -> {save_dir}/best_params.json", flush=True)

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

    print(f"  Results -> {save_dir}/", flush=True)
    return summary


def run_across_datasets(model_name, build_fn, datasets=None, is_search=False):
    """
    Run a model across all (or specified) datasets.

    Parameters
    ----------
    model_name : str
    build_fn : callable(n_features=None) -> estimator
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
            try:
                model = build_fn(X.shape[1])
            except TypeError:
                model = build_fn()
            summary = run_experiment(
                model_name, model, X, y,
                dataset_name=ds_name,
                is_search=is_search,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"\n  [ERROR] {model_name} x {ds_name}: {e}", flush=True)

    return summaries

