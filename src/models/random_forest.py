"""
Random Forest with RandomizedSearchCV — matches original Random Forest.ipynb.

Run on all datasets:  python -m src.models.random_forest
Run on one dataset:   python -m src.models.random_forest ROI_Sync
"""
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.config import RANDOM_STATE
from src.data_loader import load_dataset
from src.models.base import run_experiment, run_across_datasets, get_cv


MODEL_NAME = "RandomForest"

PARAM_DIST = {
    "clf__n_estimators": [100, 300, 500, 800],
    "clf__max_depth": [None, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2"],
    "clf__bootstrap": [True, False],
}


def build_model():
    """Build RF pipeline with RandomizedSearchCV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])
    return RandomizedSearchCV(
        pipe, param_distributions=PARAM_DIST, n_iter=15,
        cv=5, scoring="f1_weighted", refit=True,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )


def main():
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
