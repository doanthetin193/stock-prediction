"""
Prophet Model cho dự đoán giá cổ phiếu.
Model thống kê từ Meta, xử lý tốt trend + seasonality.
"""
import numpy as np
import pandas as pd
import os
import joblib
import logging

# Tắt logging verbose từ Prophet
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import PROPHET_CHANGEPOINT_PRIOR, MODELS_DIR


class ProphetModel:
    """Facebook Prophet cho time series prediction."""

    def __init__(self, changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR):
        from prophet import Prophet
        self.model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        self.is_fitted = False

    def train(self, train_df: pd.DataFrame, verbose: bool = True):
        """
        Train Prophet model.

        Args:
            train_df: DataFrame với cột 'ds' (datetime) và 'y' (giá trị)
            verbose: hiển thị thông tin
        """
        self.model.fit(train_df)
        self.is_fitted = True

        if verbose:
            print(f"  ✅ Prophet trained trên {len(train_df)} data points")

    def predict(self, periods: int = None, test_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Dự đoán giá.

        Args:
            periods: số ngày dự đoán (dùng nếu không có test_df)
            test_df: DataFrame với cột 'ds' để dự đoán

        Returns:
            DataFrame với cột ds, yhat, yhat_lower, yhat_upper
        """
        if test_df is not None:
            future = test_df[['ds']].copy()
        else:
            future = self.model.make_future_dataframe(periods=periods)

        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def get_predictions_array(self, test_df: pd.DataFrame) -> np.ndarray:
        """Trả về mảng predictions để tính metrics."""
        forecast = self.predict(test_df=test_df)
        return forecast['yhat'].values

    def save(self, symbol: str):
        """Lưu model."""
        path = os.path.join(MODELS_DIR, f"prophet_{symbol}.joblib")
        joblib.dump(self.model, path)
        print(f"  💾 Prophet model saved: {path}")

    def load(self, symbol: str):
        """Tải model đã lưu."""
        path = os.path.join(MODELS_DIR, f"prophet_{symbol}.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            print(f"  📂 Prophet model loaded: {path}")
        else:
            print(f"  ⚠️ Model file not found: {path}")
