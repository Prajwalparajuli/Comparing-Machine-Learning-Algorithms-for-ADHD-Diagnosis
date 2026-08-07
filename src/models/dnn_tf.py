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
        proba = self.model_.predict(X, verbose=0).ravel()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        proba_pos = self.model_.predict(X, verbose=0).ravel()
        return np.column_stack([1 - proba_pos, proba_pos])


def build_model():
    """Build DNN pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", TFDNNClassifier(
            hidden_units=[256, 128, 64], dropout_rate=0.3,
            epochs=150, batch_size=32, learning_rate=0.001,
            random_state=RANDOM_STATE,
        )),
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
