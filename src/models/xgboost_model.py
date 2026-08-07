"""
XGBoost with RandomizedSearchCV — matches original XGBoost.ipynb.

Run on all datasets:  python -m src.models.xgboost_model
Run on one dataset:   python -m src.models.xgboost_model FC
"""
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.config import RANDOM_STATE
from src.data_loader import load_dataset
from src.models.base import run_experiment, run_across_datasets, get_cv


MODEL_NAME = "XGBoost"

PARAM_DIST = {
    "clf__n_estimators": [100, 300, 500, 800],
    "clf__max_depth": [3, 5, 7, 10],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "clf__gamma": [0, 0.1, 0.3, 0.5],
    "clf__reg_alpha": [0, 0.01, 0.1, 1],
    "clf__reg_lambda": [0.5, 1, 2, 5],
    "clf__min_child_weight": [1, 3, 5, 7],
}


def build_model():
    """Build XGBoost pipeline with RandomizedSearchCV using CUDA GPU safely."""
    # Ensure safe CUDA GPU execution with single-worker SearchCV to prevent VRAM memory context conflicts
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            device="cuda",
            tree_method="hist",
        )),
    ])
    return RandomizedSearchCV(
        pipe, param_distributions=PARAM_DIST, n_iter=25,
        cv=get_cv(), scoring="f1_weighted", refit=True,
        random_state=RANDOM_STATE, n_jobs=1, verbose=0,
    )


def main():
    print("  [GPU Check] XGBoost set to target NVIDIA CUDA GPU (device='cuda', tree_method='hist')")
    args = sys.argv[1:]
    if args:
        for ds in args:
            X, y, _ = load_dataset(ds)
            model = build_model()
            run_experiment(MODEL_NAME, model, X, y, dataset_name=ds, is_search=True)
    else:
        run_across_datasets(MODEL_NAME, build_model, is_search=True)


if __name__ == "__main__":
    main()
