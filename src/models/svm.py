"""
SVM (RBF) with GridSearchCV — matches original SVM.ipynb.

Run on all datasets:  python -m src.models.svm
Run on one dataset:   python -m src.models.svm FC_PCA
"""
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.config import RANDOM_STATE
from src.data_loader import load_dataset
from src.models.base import run_experiment, run_across_datasets, get_cv


MODEL_NAME = "SVM_RBF"

PARAM_GRID = {
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", 0.01, 0.001],
    "clf__kernel": ["rbf"],
    "clf__shrinking": [True, False],
    "clf__tol": [1e-4, 1e-3, 1e-2],
    "clf__max_iter": [-1],
    "clf__class_weight": ["balanced"],
}


def build_model():
    """Build SVM pipeline with GridSearchCV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=RANDOM_STATE)),
    ])
    return GridSearchCV(
        pipe, param_grid=PARAM_GRID, cv=5,
        scoring="f1_weighted", refit=True, n_jobs=-1, verbose=1,
    )


def main():
    args = sys.argv[1:]
    if args:
        # Run on specific dataset(s)
        for ds in args:
            X, y, _ = load_dataset(ds)
            model = build_model()
            run_experiment(MODEL_NAME, model, X, y, dataset_name=ds, is_search=True)
    else:
        # Run on ALL datasets
        run_across_datasets(MODEL_NAME, build_model, is_search=True)


if __name__ == "__main__":
    main()
