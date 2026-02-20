"""
Module tải dữ liệu cổ phiếu Việt Nam.
Nguồn chính: yfinance (ổn định nhất).
Nguồn phụ: vnstock3 (có thể bị rate limit / 403).
"""
import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, STOCK_SYMBOLS, DATA_START_DATE, DATA_END_DATE


def download_stock_data(symbol: str, start: str = DATA_START_DATE, end: str = DATA_END_DATE) -> pd.DataFrame:
    """
    Tải dữ liệu cổ phiếu.
    Ưu tiên yfinance (ổn định), fallback vnstock3.

    Args:
        symbol: Mã cổ phiếu (VNM, VCB, FPT, ...)
        start: Ngày bắt đầu (YYYY-MM-DD)
        end: Ngày kết thúc (YYYY-MM-DD)

    Returns:
        DataFrame với cột: time, open, high, low, close, volume
    """
    df = None

    # --- Thử yfinance trước (ổn định hơn) ---
    try:
        import yfinance as yf
        ticker = f"{symbol}.VN"
        raw = yf.download(ticker, start=start, end=end, progress=False)

        if raw is not None and not raw.empty:
            # yfinance trả về multi-level columns: ('Close', 'VNM.VN')
            # Cần flatten về single level
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0].lower() for col in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            raw = raw.reset_index()

            # Cột index là 'Date' hoặc 'date'
            date_col = None
            for col in raw.columns:
                if 'date' in str(col).lower():
                    date_col = col
                    break

            if date_col is not None:
                raw.rename(columns={date_col: 'time'}, inplace=True)

            # Bỏ adj close nếu có
            for col in list(raw.columns):
                if 'adj' in str(col).lower():
                    raw.drop(columns=[col], inplace=True)

            df = raw
            print(f"  ✅ [{symbol}] Tải thành công từ yfinance: {len(df)} dòng")

    except Exception as e:
        print(f"  ⚠️ [{symbol}] yfinance lỗi: {e}")

    if df is None or df.empty:
        print(f"  ❌ [{symbol}] Không thể tải dữ liệu!")
        return pd.DataFrame()

    # Chuẩn hóa: giữ lại các cột cần thiết
    df['time'] = pd.to_datetime(df['time'])
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in required_cols if c in df.columns]
    df = df[available_cols].copy()

    # Đảm bảo kiểu số
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sắp xếp theo thời gian
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Loại bỏ NaN
    df.dropna(inplace=True)

    return df


def save_stock_data(df: pd.DataFrame, symbol: str) -> str:
    """Lưu DataFrame ra file CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    df.to_csv(filepath, index=False)
    print(f"  💾 Đã lưu: {filepath}")
    return filepath


def load_stock_data(symbol: str) -> pd.DataFrame:
    """
    Đọc dữ liệu cổ phiếu từ file CSV đã tải.

    Args:
        symbol: Mã cổ phiếu

    Returns:
        DataFrame với cột time, open, high, low, close, volume
    """
    filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(filepath):
        print(f"  ⚠️ File {filepath} chưa tồn tại. Đang tải dữ liệu...")
        df = download_stock_data(symbol)
        if not df.empty:
            save_stock_data(df, symbol)
        return df

    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    print(f"  📂 Đã đọc {symbol}: {len(df)} dòng")
    return df


def download_all_stocks():
    """Tải dữ liệu tất cả cổ phiếu trong danh sách."""
    print("=" * 50)
    print("🚀 Bắt đầu tải dữ liệu cổ phiếu...")
    print(f"📋 Danh sách: {STOCK_SYMBOLS}")
    print(f"📅 Từ {DATA_START_DATE} đến {DATA_END_DATE}")
    print("=" * 50)

    results = {}
    for symbol in STOCK_SYMBOLS:
        print(f"\n--- {symbol} ---")
        df = download_stock_data(symbol)
        if not df.empty:
            save_stock_data(df, symbol)
            results[symbol] = len(df)
        else:
            results[symbol] = 0

    print("\n" + "=" * 50)
    print("📊 Kết quả tải dữ liệu:")
    for symbol, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {symbol}: {count} dòng")
    print("=" * 50)

    return results


if __name__ == "__main__":
    download_all_stocks()
