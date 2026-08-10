"""
DNN (TensorFlow/Keras) — feed-forward for flat feature vectors.

Run on all datasets:  python -m src.models.dnn_tf
Run on one dataset:   python -m src.models.dnn_tf Combined
"""
import sys
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.data_loader import load_dataset
from src.models.base import run_experiment, run_across_datasets


MODEL_NAME = "DNN_TF"


def setup_tf_gpu():
    """Configure TensorFlow environment and GPU settings."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  [GPU Check] TensorFlow CUDA GPU active: {len(gpus)} GPU(s) ({gpus[0].name})", flush=True)
        except Exception as e:
            print(f"  [GPU Check] TensorFlow GPU note: {e}", flush=True)
    else:
        print("  [GPU Check] TensorFlow active (CPU / default compute device)", flush=True)


class TFDNNClassifier(BaseEstimator, ClassifierMixin):
    """TensorFlow/Keras DNN wrapped as sklearn estimator."""

    def __init__(self, hidden_units=None, dropout_rate=0.3,
                 epochs=150, batch_size=32, learning_rate=0.001,
                 random_state=42):
        self.hidden_units = hidden_units or [256, 128, 64]
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _build_model(self, n_features):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(n_features,)))
        for units in self.hidden_units:
            model.add(tf.keras.layers.Dense(
                units, activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        self.classes_ = np.unique(y)
        self.model_ = self._build_model(X.shape[1])

        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=self.classes_, y=y)
        class_weight = dict(zip(self.classes_, cw))

        self.model_.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            class_weight=class_weight, validation_split=0.15,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True)],
            verbose=0,
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        proba = self.model_.predict(X, verbose=0).ravel()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        proba_pos = self.model_.predict(X, verbose=0).ravel()
        return np.column_stack([1 - proba_pos, proba_pos])


from sklearn.feature_selection import SelectKBest, f_classif


def build_model(n_features=None):
    """Build DNN pipeline."""
    steps = [("scaler", StandardScaler())]
    if n_features and n_features > 2000:
        steps.append(("select", SelectKBest(f_classif, k=1000)))
    steps.append(("clf", TFDNNClassifier(
        hidden_units=[256, 128, 64], dropout_rate=0.3,
        epochs=150, batch_size=32, learning_rate=0.001,
        random_state=RANDOM_STATE,
    )))
    return Pipeline(steps)


def main():
    setup_tf_gpu()
    args = sys.argv[1:]
    if args:
        for ds in args:
            X, y, _ = load_dataset(ds)
            model = build_model(X.shape[1])
            run_experiment(MODEL_NAME, model, X, y, dataset_name=ds, is_search=False)
    else:
        run_across_datasets(MODEL_NAME, build_model, is_search=False)


if __name__ == "__main__":
    main()

