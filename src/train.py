"""
Orchestrator — runs ALL models × ALL datasets.

Usage:
  python -m src.train                    # All models × all datasets
  python -m src.train FC_PCA Combined    # All models × specific datasets

Each model can also run independently:
  python -m src.models.svm               # SVM × all datasets
  python -m src.models.svm FC_PCA        # SVM × FC_PCA only
"""
import sys
import pandas as pd
from src.config import RESULTS_DIR
from src.data_loader import DATASET_NAMES

# Import each model's build function and name
from src.models.dummy_baseline import build_model as build_dummy, MODEL_NAME as DUMMY_NAME
from src.models.svm import build_model as build_svm, MODEL_NAME as SVM_NAME
from src.models.random_forest import build_model as build_rf, MODEL_NAME as RF_NAME
from src.models.decision_tree import build_model as build_dt, MODEL_NAME as DT_NAME
from src.models.xgboost_model import build_model as build_xgb, MODEL_NAME as XGB_NAME

from src.models.base import run_across_datasets

# Optional TF model
try:
    from src.models.dnn_tf import build_model as build_dnn, MODEL_NAME as DNN_NAME
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARN] TensorFlow not available — skipping DNN_TF.")


MODEL_REGISTRY = [
    (DUMMY_NAME,  build_dummy,  False),   # (name, builder, is_search)
    (SVM_NAME,    build_svm,    True),
    (RF_NAME,     build_rf,     True),
    (DT_NAME,     build_dt,     True),
    (XGB_NAME,    build_xgb,    True),
]

if HAS_TF:
    MODEL_REGISTRY.append((DNN_NAME, build_dnn, False))


def main():
    # Parse optional dataset filter from CLI
    args = sys.argv[1:]
    datasets = args if args else None  # None = all

    ds_label = ", ".join(datasets) if datasets else "ALL"
    print(f"\n{'='*60}")
    print(f"  ADHD ML Pipeline — {len(MODEL_REGISTRY)} models × {ds_label} datasets")
    print(f"{'='*60}")

    all_summaries = []

    for model_name, build_fn, is_search in MODEL_REGISTRY:
        try:
            summaries = run_across_datasets(
                model_name, build_fn,
                datasets=datasets, is_search=is_search,
            )
            all_summaries.extend(summaries)
        except Exception as e:
            print(f"\n  [ERROR] {model_name} failed: {e}")

    # Save combined summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_summaries)
    df.to_csv(RESULTS_DIR / "summary_all.csv", index=False)

    # Print results matrix
    print(f"\n{'='*60}")
    print(f"  RESULTS MATRIX (Accuracy mean ± std)")
    print(f"{'='*60}")

    if not df.empty:
        pivot = df.pivot_table(
            index="model", columns="dataset",
            values="acc_mean", aggfunc="first"
        )
        print(pivot.to_string(float_format="{:.4f}".format))

    print(f"\n  Full results → {RESULTS_DIR / 'summary_all.csv'}")
    print(f"  Per-model results → results/{{MODEL}}/{{DATASET}}/")


if __name__ == "__main__":
    main()
