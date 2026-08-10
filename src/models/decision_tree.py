"""
Decision Tree with RandomizedSearchCV — matches original Decision Tree.ipynb.

Run on all datasets:  python -m src.models.decision_tree
Run on one dataset:   python -m src.models.decision_tree fALFF
"""
import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.config import RANDOM_STATE
from src.data_loader import load_dataset
from src.models.base import run_experiment, run_across_datasets, get_cv


MODEL_NAME = "DecisionTree"

PARAM_DIST = {
    "clf__max_depth": [None, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", None],
    "clf__criterion": ["gini", "entropy"],
    "clf__ccp_alpha": np.linspace(0.0, 0.02, 10),
}


def build_model():
    """Build DT pipeline with RandomizedSearchCV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
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
