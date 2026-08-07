"""
Statistical evaluation — Bootstrap CIs, McNemar's tests.

Reads per-model/per-dataset results from results/{MODEL}/{DATASET}/
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import (
    RESULTS_DIR, MODEL_NAMES, N_BOOTSTRAP, CI_ALPHA, RANDOM_STATE,
)
from src.data_loader import DATASET_NAMES


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP,
                  alpha=CI_ALPHA, rng=None):
    """Compute bootstrapped confidence interval for a metric."""
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(lo), float(hi)


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test comparing two classifiers."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p_value)


def discover_results():
    """Scan results/ for model/dataset directories with OOF predictions."""
    found = {}
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            oof_path = ds_dir / "oof_predictions.csv"
            if oof_path.exists():
                ds_name = ds_dir.name
                if ds_name not in found:
                    found[ds_name] = {}
                df = pd.read_csv(oof_path)
                found[ds_name][model_name] = {
                    "y_true": df["y_true"].values,
                    "y_pred": df["y_pred"].values,
                    "y_proba": df["y_proba"].values if "y_proba" in df.columns else None,
                }
    return found


def main():
    print("=" * 60)
    print("  ADHD ML Pipeline — Statistical Evaluation")
    print("=" * 60)

    all_results = discover_results()

    if not all_results:
        print("  [ERROR] No results found. Run training first.")
        return

    all_ci = []
    all_mcnemar = []

    for ds_name in all_results:
        models = all_results[ds_name]
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name} ({len(models)} models)")
        print(f"{'─'*60}")

        rng = np.random.RandomState(RANDOM_STATE)
        ci_results = []

        for model_name, data in models.items():
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            y_proba = data["y_proba"]

            acc_m, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score, rng=rng)
            f1_m, f1_lo, f1_hi = bootstrap_ci(
                y_true, y_pred,
                lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
                rng=rng)

            if y_proba is not None and not np.all(np.isnan(y_proba)):
                auc_m, auc_lo, auc_hi = bootstrap_ci(
                    y_true, y_proba, lambda yt, yp: roc_auc_score(yt, yp), rng=rng)
            else:
                auc_m = auc_lo = auc_hi = np.nan

            row = {
                "dataset": ds_name, "model": model_name,
                "acc_mean": acc_m, "acc_lo": acc_lo, "acc_hi": acc_hi,
                "f1_mean": f1_m, "f1_lo": f1_lo, "f1_hi": f1_hi,
                "auc_mean": auc_m, "auc_lo": auc_lo, "auc_hi": auc_hi,
            }
            ci_results.append(row)
            all_ci.append(row)

            print(f"\n  {model_name}:")
            print(f"    Acc:  {acc_m:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
            print(f"    F1:   {f1_m:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")
            if not np.isnan(auc_m):
                print(f"    AUC:  {auc_m:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

        # Best model for this dataset
        non_dummy = [r for r in ci_results if r["model"] != "DummyBaseline"]
        if non_dummy:
            best = max(non_dummy, key=lambda r: r["f1_mean"])
            best_name = best["model"]
            print(f"\n  Best ({ds_name}): {best_name} F1={best['f1_mean']:.4f}")

            y_pred_best = models[best_name]["y_pred"]
            y_true = models[best_name]["y_true"]

            for model_name, data in models.items():
                if model_name == best_name:
                    continue
                stat, p = mcnemar_test(y_true, y_pred_best, data["y_pred"])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                all_mcnemar.append({
                    "dataset": ds_name,
                    "model_a": best_name, "model_b": model_name,
                    "chi2": stat, "p_value": p, "significant": sig,
                })

    # Save
    ci_df = pd.DataFrame(all_ci)
    ci_df.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)

    mcnemar_df = pd.DataFrame(all_mcnemar)
    mcnemar_df.to_csv(RESULTS_DIR / "mcnemar_tests.csv", index=False)

    combined = {
        "bootstrap_ci": all_ci,
        "mcnemar_tests": all_mcnemar,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_alpha": CI_ALPHA,
    }
    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n  All results saved to {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
