"""
Central configuration for the ADHD ML comparison pipeline.
All paths, hyperparameters, and constants in one place.
"""
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

# Create output dirs
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Data ───────────────────────────────────────────────────────
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
META_PATH = DATA_DIR / "metadata.csv"

# ─── Cross-Validation ──────────────────────────────────────────
N_FOLDS = 10
RANDOM_STATE = 42

# ─── Bootstrap CI ──────────────────────────────────────────────
N_BOOTSTRAP = 2000
CI_ALPHA = 0.05  # 95% CI

# ─── Model Names (canonical order — matches original notebooks) ─
MODEL_NAMES = [
    "DummyBaseline",
    "SVM_RBF",
    "RandomForest",
    "DecisionTree",
    "XGBoost",
    "DNN_TF",
]

# ─── Visualization ─────────────────────────────────────────────
FIGSIZE_WIDE = (14, 7)
FIGSIZE_SQUARE = (8, 8)
DPI = 300
FONT_FAMILY = "serif"

# Color palette — curated for publication
PALETTE = {
    "DummyBaseline":  "#9e9e9e",
    "SVM_RBF":        "#ef5350",
    "RandomForest":   "#66bb6a",
    "DecisionTree":   "#8d6e63",
    "XGBoost":        "#ffa726",
    "DNN_TF":         "#42a5f5",
}
