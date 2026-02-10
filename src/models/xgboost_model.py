"""
XGBoost Model cho dự đoán giá cổ phiếu.
Sử dụng tabular features (OHLCV + Technical Indicators).
"""
import numpy as np
import os
import joblib
from xgboost import XGBRegressor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, MODELS_DIR


class XGBoostModel:
    """XGBoost Regressor cho stock price prediction."""

    def __init__(self, n_estimators: int = XGB_N_ESTIMATORS,
                 max_depth: int = XGB_MAX_DEPTH,
                 learning_rate: float = XGB_LEARNING_RATE):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        self.feature_names = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: list = None, verbose: bool = True):
        """
        Train XGBoost model.

        Args:
            X_train: shape (samples, features)
            y_train: shape (samples,)
            X_val: validation features (optional)
            y_val: validation target (optional)
            feature_names: tên các features
            verbose: hiển thị tiến trình
        """
        self.feature_names = feature_names

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )

        if verbose:
            print(f"  ✅ XGBoost trained: {self.model.n_estimators} trees")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán giá."""
        return self.model.predict(X)

    def get_feature_importance(self) -> dict:
        """Lấy feature importance."""
        importance = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return dict(enumerate(importance))

    def save(self, symbol: str):
        """Lưu model."""
        path = os.path.join(MODELS_DIR, f"xgboost_{symbol}.joblib")
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)
        print(f"  💾 XGBoost model saved: {path}")

    def load(self, symbol: str):
        """Tải model đã lưu."""
        path = os.path.join(MODELS_DIR, f"xgboost_{symbol}.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.feature_names = data['feature_names']
            print(f"  📂 XGBoost model loaded: {path}")
        else:
            print(f"  ⚠️ Model file not found: {path}")
