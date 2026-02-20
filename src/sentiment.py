"""
Module Sentiment Analysis cho tin tức tài chính Việt Nam.
- Market-based sentiment: tính từ dữ liệu giá (luôn hoạt động)
- News-based sentiment: crawl tin tức từ CafeF/VnExpress (cần bs4, requests)
"""
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


# ============================================================
# Crawl Tin Tức
# ============================================================

def crawl_cafef_news(symbol: str, max_pages: int = 3) -> list:
    """
    Crawl tin tức từ CafeF liên quan đến mã cổ phiếu.

    Args:
        symbol: Mã cổ phiếu (VNM, FPT, ...)
        max_pages: Số trang crawl

    Returns:
        List of dict: [{'title': ..., 'date': ..., 'content': ...}]
    """
    articles = []
    import requests
    from bs4 import BeautifulSoup
    import time
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for page in range(1, max_pages + 1):
        try:
            url = f"https://cafef.vn/tim-kiem.chn?keywords={symbol}&page={page}"
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # Tìm các bài viết
            items = soup.find_all('div', class_='tlitem') or soup.find_all('li', class_='news-item')

            for item in items:
                try:
                    title_tag = item.find('a', class_='title') or item.find('h3')
                    if title_tag:
                        title = title_tag.get_text(strip=True)

                        # Lấy ngày
                        date_tag = item.find('span', class_='time') or item.find('time')
                        date_str = date_tag.get_text(strip=True) if date_tag else ""

                        articles.append({
                            'title': title,
                            'date': date_str,
                            'symbol': symbol
                        })
                except Exception:
                    continue

            print(f"  📰 CafeF page {page}: {len(items)} bài viết")
            if page < max_pages:
                time.sleep(1)  # Tránh bị block

        except Exception as e:
            print(f"  ⚠️ Lỗi crawl CafeF page {page}: {e}")

    return articles


def crawl_vnexpress_news(symbol: str, max_pages: int = 3) -> list:
    """
    Crawl tin tức từ VnExpress phần kinh doanh/chứng khoán.

    Args:
        symbol: Mã cổ phiếu
        max_pages: Số trang crawl

    Returns:
        List of dict
    """
    articles = []
    import requests
    from bs4 import BeautifulSoup
    import time
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    for page in range(1, max_pages + 1):
        try:
            url = f"https://timkiem.vnexpress.net/?q={symbol}&cate_code=kinhdoanh&page={page}"
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')
            items = soup.find_all('article', class_='item-news')

            for item in items:
                try:
                    title_tag = item.find('h3') or item.find('h2')
                    if title_tag:
                        a_tag = title_tag.find('a')
                        title = a_tag.get_text(strip=True) if a_tag else title_tag.get_text(strip=True)

                        desc_tag = item.find('p', class_='description')
                        desc = desc_tag.get_text(strip=True) if desc_tag else ""

                        date_tag = item.find('span', class_='time-ago')
                        date_str = date_tag.get_text(strip=True) if date_tag else ""

                        articles.append({
                            'title': title,
                            'description': desc,
                            'date': date_str,
                            'symbol': symbol
                        })
                except Exception:
                    continue

            print(f"  📰 VnExpress page {page}: {len(items)} bài viết")
            if page < max_pages:
                time.sleep(1)  # Tránh bị block

        except Exception as e:
            print(f"  ⚠️ Lỗi crawl VnExpress page {page}: {e}")

    return articles


# ============================================================
# Sentiment Analysis
# ============================================================

def analyze_sentiment_textblob(text: str) -> float:
    """
    Phân tích sentiment bằng TextBlob.
    Đơn giản, hoạt động tốt với tiếng Anh, OK cho tiếng Việt.

    Returns:
        Score từ -1 (tiêu cực) đến +1 (tích cực)
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception:
        return 0.0


def analyze_sentiment_vietnamese(text: str) -> float:
    """
    Phân tích sentiment cho tiếng Việt bằng từ điển.
    Đơn giản nhưng hiệu quả cho tin tức tài chính.

    Returns:
        Score từ -1 đến +1
    """
    # Từ điển sentiment cơ bản cho tài chính Việt Nam
    positive_words = [
        'tăng', 'lãi', 'tích cực', 'tốt', 'tăng trưởng', 'kỷ lục',
        'đột phá', 'vượt', 'thuận lợi', 'khả quan', 'bứt phá',
        'hồi phục', 'triển vọng', 'lạc quan', 'đà tăng', 'điểm sáng',
        'cải thiện', 'hiệu quả', 'thặng dư', 'dẫn đầu', 'bền vững',
        'phát triển', 'mở rộng', 'doanh thu tăng', 'lợi nhuận tăng'
    ]

    negative_words = [
        'giảm', 'lỗ', 'tiêu cực', 'xấu', 'sụt giảm', 'rủi ro',
        'khó khăn', 'tụt', 'bất lợi', 'lo ngại', 'suy yếu',
        'đáy', 'đổ vỡ', 'bi quan', 'đà giảm', 'cảnh báo',
        'nợ xấu', 'thua lỗ', 'phá sản', 'khủng hoảng', 'sụp đổ',
        'bán tháo', 'lao dốc', 'thâm hụt', 'doanh thu giảm'
    ]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    score = (pos_count - neg_count) / total
    return round(score, 4)


def get_sentiment_for_stock(symbol: str, use_vietnamese: bool = True) -> pd.DataFrame:
    """
    Crawl tin tức và tính sentiment score cho mã cổ phiếu.

    Args:
        symbol: Mã cổ phiếu
        use_vietnamese: True = dùng từ điển tiếng Việt, False = dùng TextBlob

    Returns:
        DataFrame (title, date, sentiment_score)
    """
    print(f"\n🔍 Crawl & phân tích sentiment cho {symbol}...")

    # Crawl tin tức
    articles = crawl_cafef_news(symbol)
    articles.extend(crawl_vnexpress_news(symbol))

    if not articles:
        print(f"  ⚠️ Không tìm được tin tức cho {symbol}")
        return pd.DataFrame(columns=['title', 'date', 'sentiment_score'])

    # Phân tích sentiment
    results = []
    for article in articles:
        text = article.get('title', '') + ' ' + article.get('description', '')

        if use_vietnamese:
            score = analyze_sentiment_vietnamese(text)
        else:
            score = analyze_sentiment_textblob(text)

        results.append({
            'title': article.get('title', ''),
            'date': article.get('date', ''),
            'sentiment_score': score
        })

    df = pd.DataFrame(results)

    # Thống kê
    avg_score = df['sentiment_score'].mean()
    sentiment_label = "Tích cực 📈" if avg_score > 0 else "Tiêu cực 📉" if avg_score < 0 else "Trung lập ➡️"

    print(f"  📊 Tổng: {len(df)} tin tức")
    print(f"  📊 Sentiment trung bình: {avg_score:.4f} ({sentiment_label})")

    return df


def save_sentiment_data(df: pd.DataFrame, symbol: str) -> str:
    """Lưu sentiment data ra CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{symbol}_sentiment.csv")
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"  💾 Đã lưu sentiment: {filepath}")
    return filepath


# ============================================================
# Market-based Sentiment (tính từ dữ liệu giá)
# ============================================================

def compute_market_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính market-based sentiment từ dữ liệu giá cổ phiếu.
    Wam dụng ý tưởng: giá phản ánh tâm lý thị trường.

    Tạo 3 features:
    - sentiment_momentum: dựa trên xu hướng giá 5 ngày gần nhất
    - sentiment_volatility: dựa trên biến động giá (cao = bất ổn = tiêu cực)
    - sentiment_score: tổng hợp = momentum * 0.6 + volume_signal * 0.2 - volatility * 0.2

    Returns:
        DataFrame gốc + 3 cột sentiment mới
    """
    df = df.copy()

    # --- 1. Momentum Sentiment (xu hướng giá 5 ngày) ---
    # Return 5 ngày: > 0 = tích cực, < 0 = tiêu cực
    returns_5d = df['close'].pct_change(5)
    # Chuẩn hóa về [-1, 1] bằng tanh (sigmoid-like, smooth)
    df['sentiment_momentum'] = np.tanh(returns_5d * 10)

    # --- 2. Volatility Sentiment (biến động = bất ổn) ---
    # Std of returns trong 10 ngày, chuẩn hóa
    daily_returns = df['close'].pct_change()
    rolling_vol = daily_returns.rolling(10).std()
    # Chuẩn hóa: vol cao → sentiment thấp (bất ổn = tiêu cực)
    vol_median = rolling_vol.median()
    if vol_median > 0:
        vol_normalized = rolling_vol / (vol_median * 3)  # scale
        df['sentiment_volatility'] = vol_normalized.clip(0, 1)
    else:
        df['sentiment_volatility'] = 0.0

    # --- 3. Volume Signal ---
    # Volume tăng đột biến khi có tin tức/sự kiện
    vol_sma20 = df['volume'].rolling(20).mean()
    volume_ratio = df['volume'] / vol_sma20
    # Volume > 1.5x trung bình = sự kiện, nhân với hướng giá
    volume_signal = np.tanh((volume_ratio - 1) * 2) * np.sign(daily_returns)

    # --- 4. Tổng hợp Sentiment Score ---
    df['sentiment_score'] = (
        df['sentiment_momentum'] * 0.6 +
        volume_signal.fillna(0) * 0.2 -
        df['sentiment_volatility'] * 0.2
    ).round(4)

    # Clip về [-1, 1]
    df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)

    # Fill NaN (từ rolling) bằng 0 (neutral)
    for col in ['sentiment_momentum', 'sentiment_volatility', 'sentiment_score']:
        df[col] = df[col].fillna(0)

    print(f"  📊 Market Sentiment: mean={df['sentiment_score'].mean():.4f}, "
          f"positive={( df['sentiment_score'] > 0).sum()}, "
          f"negative={(df['sentiment_score'] < 0).sum()}")

    return df


def merge_sentiment_with_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Merge sentiment vào DataFrame cổ phiếu.
    - Market-based sentiment: luôn có (tính từ giá)
    - News-based sentiment: override nếu có file CSV đã crawl

    Args:
        df: DataFrame cổ phiếu gốc (cần cột 'time', 'close', 'volume')
        symbol: mã cổ phiếu

    Returns:
        DataFrame + 3 cột: sentiment_score, sentiment_momentum, sentiment_volatility
    """
    # Bước 1: Tính market-based sentiment cho toàn bộ lịch sử
    df = compute_market_sentiment(df)

    # Bước 2: Override bằng news-based nếu có
    news_file = os.path.join(DATA_DIR, f"{symbol}_sentiment.csv")
    if os.path.exists(news_file):
        try:
            news_df = pd.read_csv(news_file)
            if 'date' in news_df.columns and 'sentiment_score' in news_df.columns:
                # Parse ngày từ news data
                news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
                news_df = news_df.dropna(subset=['date'])

                if not news_df.empty:
                    # Aggregate: trung bình sentiment theo ngày
                    daily_news = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].mean()

                    # Override market-based bằng news-based cho những ngày có tin
                    df_dates = pd.to_datetime(df['time']).dt.date
                    for date_val, score in daily_news.items():
                        mask = df_dates == date_val
                        if mask.any():
                            # Kết hợp: 50% market + 50% news
                            market_score = df.loc[mask, 'sentiment_score'].values[0]
                            blended = market_score * 0.5 + score * 0.5
                            df.loc[mask, 'sentiment_score'] = round(blended, 4)

                    n_overridden = sum(1 for d in daily_news.index if (df_dates == d).any())
                    print(f"  📰 News sentiment merged: {n_overridden} ngày được cập nhật từ tin tức")

        except Exception as e:
            print(f"  ⚠️ Không thể merge news sentiment: {e}")

    return df


if __name__ == "__main__":
    # Demo: crawl sentiment cho VNM
    df = get_sentiment_for_stock("VNM")
    if not df.empty:
        save_sentiment_data(df, "VNM")
        print(df.head(10))

