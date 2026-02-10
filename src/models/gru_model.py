"""
GRU Model cho dự đoán giá cổ phiếu.
Kiến trúc: 2 lớp GRU + Dropout + Dense
Nhẹ hơn LSTM, ít tham số hơn.
"""
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import DL_EPOCHS, DL_BATCH_SIZE, DL_LEARNING_RATE, MODELS_DIR


class GRUModel:
    """GRU Model cho time series prediction."""

    def __init__(self, seq_length: int, n_features: int,
                 epochs: int = DL_EPOCHS, batch_size: int = DL_BATCH_SIZE,
                 learning_rate: float = DL_LEARNING_RATE):
        self.seq_length = seq_length
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> Sequential:
        """Xây dựng kiến trúc GRU."""
        model = Sequential([
            GRU(128, return_sequences=True,
                input_shape=(self.seq_length, self.n_features)),
            Dropout(0.2),

            GRU(64, return_sequences=False),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose: int = 1):
        """Train model."""
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                          patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                              factor=0.5, patience=5, min_lr=1e-6)
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán giá."""
        return self.model.predict(X, verbose=0).flatten()

    def save(self, symbol: str):
        """Lưu model."""
        path = os.path.join(MODELS_DIR, f"gru_{symbol}.keras")
        self.model.save(path)
        print(f"  💾 GRU model saved: {path}")

    def load(self, symbol: str):
        """Tải model đã lưu."""
        path = os.path.join(MODELS_DIR, f"gru_{symbol}.keras")
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"  📂 GRU model loaded: {path}")
        else:
            print(f"  ⚠️ Model file not found: {path}")

    def summary(self):
        """In kiến trúc model."""
        return self.model.summary()
