"""
Dummy baseline — majority-class classifier.

Run on all datasets:  python -m src.models.dummy_baseline
Run on one dataset:   python -m src.models.dummy_baseline FC_PCA
"""
import sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from src.config import RANDOM_STATE
from src.data_loader import load_dataset
from src.models.base import run_experiment, run_across_datasets


MODEL_NAME = "DummyBaseline"


def build_model():
    """Majority-class baseline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DummyClassifier(strategy="most_frequent",
                                 random_state=RANDOM_STATE)),
    ])


def main():
    args = sys.argv[1:]
    if args:
        for ds in args:
            X, y, _ = load_dataset(ds)
            model = build_model()
            run_experiment(MODEL_NAME, model, X, y, dataset_name=ds, is_search=False)
    else:
        run_across_datasets(MODEL_NAME, build_model, is_search=False)


if __name__ == "__main__":
    main()
