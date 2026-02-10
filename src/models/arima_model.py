"""
ARIMA Model cho dự đoán giá cổ phiếu.
Sử dụng auto_arima từ pmdarima để tự động tìm tham số (p, d, q).
"""
import numpy as np
import pandas as pd
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODELS_DIR


class ARIMAModel:
    """ARIMA / Auto-ARIMA cho time series prediction."""

    def __init__(self):
        self.model = None
        self.order = None
        self.is_fitted = False

    def train(self, train_series: pd.Series, verbose: bool = True):
        """
        Train ARIMA model bằng auto_arima.

        Args:
            train_series: Chuỗi thời gian close price
            verbose: hiển thị thông tin
        """
        import pmdarima as pm

        if verbose:
            print("  🔍 Đang tìm tham số ARIMA tối ưu (auto_arima)...")

        self.model = pm.auto_arima(
            train_series,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            d=None,          # Tự động tìm d
            seasonal=False,   # Không dùng seasonal cho daily stock
            stepwise=True,    # Nhanh hơn
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )

        self.order = self.model.order
        self.is_fitted = True

        if verbose:
            print(f"  ✅ ARIMA trained: order={self.order}")
            print(f"     AIC={self.model.aic():.2f}")

    def predict(self, n_periods: int) -> np.ndarray:
        """
        Dự đoán n_periods ngày tiếp theo.

        Args:
            n_periods: số ngày cần dự đoán

        Returns:
            Mảng predictions
        """
        if not self.is_fitted:
            raise ValueError("Model chưa được train!")

        predictions = self.model.predict(n_periods=n_periods)
        return np.array(predictions)

    def predict_with_confidence(self, n_periods: int):
        """Dự đoán kèm khoảng tin cậy."""
        predictions, conf_int = self.model.predict(
            n_periods=n_periods, return_conf_int=True
        )
        return np.array(predictions), conf_int

    def save(self, symbol: str):
        """Lưu model."""
        path = os.path.join(MODELS_DIR, f"arima_{symbol}.joblib")
        joblib.dump({
            'model': self.model,
            'order': self.order
        }, path)
        print(f"  💾 ARIMA model saved: {path}")

    def load(self, symbol: str):
        """Tải model đã lưu."""
        path = os.path.join(MODELS_DIR, f"arima_{symbol}.joblib")
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.order = data['order']
            self.is_fitted = True
            print(f"  📂 ARIMA model loaded: {path} (order={self.order})")
        else:
            print(f"  ⚠️ Model file not found: {path}")
