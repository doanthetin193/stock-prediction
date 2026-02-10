"""
Module tiền xử lý dữ liệu cho các models.
- Thêm Technical Indicators (SMA, EMA, RSI, MACD)
- Chuẩn bị data cho DL (LSTM/GRU), ML (XGBoost), Prophet, ARIMA
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEQUENCE_LENGTH, TEST_RATIO, FEATURE_COLUMNS, TARGET_COLUMN


# ============================================================
# Technical Indicators
# ============================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame.

    Thêm: SMA_10, SMA_20, SMA_50, EMA_12, EMA_26, RSI_14,
          MACD, MACD_Signal, Price_Change, Price_Change_Pct
    """
    df = df.copy()

    # --- Simple Moving Average ---
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # --- Exponential Moving Average ---
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # --- RSI (Relative Strength Index) ---
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- MACD ---
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # --- Price Change ---
    df['price_change'] = df['close'].diff()
    df['price_change_pct'] = df['close'].pct_change() * 100

    # --- Volume Change ---
    df['volume_change_pct'] = df['volume'].pct_change() * 100

    # Loại bỏ NaN từ rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# Data cho Deep Learning (LSTM / GRU)
# ============================================================

def create_sequences(data: np.ndarray, seq_length: int = SEQUENCE_LENGTH):
    """
    Tạo sequences cho LSTM/GRU.

    Args:
        data: Mảng 2D (samples, features) đã được scale
        seq_length: Số time steps để lookback

    Returns:
        X: shape (samples, seq_length, features)
        y: shape (samples,) — giá trị close tại time step tiếp theo
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])     # seq_length ngày trước
        y.append(data[i, 0])                 # Close ở vị trí 0 (sau reorder)
    return np.array(X), np.array(y)


def prepare_data_dl(df: pd.DataFrame, seq_length: int = SEQUENCE_LENGTH,
                    test_ratio: float = TEST_RATIO):
    """
    Chuẩn bị dữ liệu cho LSTM/GRU.

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Chọn features — đặt close lên đầu
    feature_cols = ['close', 'open', 'high', 'low', 'volume']
    # Thêm indicators nếu có
    extra_cols = ['sma_10', 'sma_20', 'ema_12', 'rsi_14', 'macd']
    for col in extra_cols:
        if col in df.columns:
            feature_cols.append(col)

    data = df[feature_cols].values

    # Scale dữ liệu về [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Tạo sequences
    X, y = create_sequences(data_scaled, seq_length)

    # Chia train / test
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  📊 DL Data: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"     Features: {feature_cols}")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# Data cho Machine Learning (XGBoost)
# ============================================================

def prepare_data_ml(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    """
    Chuẩn bị dữ liệu cho XGBoost.
    Dùng technical indicators làm features, close ngày tiếp theo làm target.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # Features cho ML
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    extra_cols = ['sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                  'rsi_14', 'macd', 'macd_signal', 'price_change',
                  'price_change_pct', 'volume_change_pct']
    for col in extra_cols:
        if col in df.columns:
            feature_cols.append(col)

    # Target: giá close ngày tiếp theo
    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    X = df[feature_cols].values
    y = df['target'].values

    # Chia train / test
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  📊 ML Data: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"     Features ({len(feature_cols)}): {feature_cols}")

    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================
# Data cho Prophet
# ============================================================

def prepare_data_prophet(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    """
    Chuẩn bị dữ liệu cho Prophet.
    Prophet yêu cầu cột 'ds' (datetime) và 'y' (giá trị).

    Returns:
        train_df, test_df
    """
    prophet_df = pd.DataFrame({
        'ds': df['time'],
        'y': df['close']
    })

    split = int(len(prophet_df) * (1 - test_ratio))
    train_df = prophet_df[:split].copy()
    test_df = prophet_df[split:].copy()

    print(f"  📊 Prophet Data: train={len(train_df)}, test={len(test_df)}")

    return train_df, test_df


# ============================================================
# Data cho ARIMA
# ============================================================

def prepare_data_arima(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    """
    Chuẩn bị dữ liệu cho ARIMA.
    ARIMA cần chuỗi thời gian đơn biến (close price).

    Returns:
        train_series, test_series
    """
    close_series = df.set_index('time')['close']

    split = int(len(close_series) * (1 - test_ratio))
    train_series = close_series[:split]
    test_series = close_series[split:]

    print(f"  📊 ARIMA Data: train={len(train_series)}, test={len(test_series)}")

    return train_series, test_series


# ============================================================
# Inverse transform — chuyển từ giá trị scale về giá trị thực
# ============================================================

def inverse_transform_predictions(predictions: np.ndarray, scaler: MinMaxScaler,
                                   n_features: int) -> np.ndarray:
    """
    Chuyển predictions từ scaled về giá trị thực.
    Vì scaler được fit trên nhiều features, cần padding.

    Args:
        predictions: mảng 1D predictions đã scale
        scaler: MinMaxScaler đã fit
        n_features: số features ban đầu

    Returns:
        Mảng 1D giá trị thực
    """
    # Tạo mảng dummy với n_features cột, đặt predictions vào cột 0 (close)
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, 0] = predictions
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, 0]
