# 📈 Dự đoán Giá Cổ phiếu Việt Nam

> **Môn học:** Lập trình Trí tuệ Nhân tạo  
> **Yêu cầu:** Đề tài có yếu tố AI (Machine Learning / Deep Learning / các kỹ thuật hiện đại)

---

## Tổng quan
Xây dựng hệ thống dự đoán giá cổ phiếu VN sử dụng nhiều mô hình ML/DL, kết hợp Sentiment Analysis tích hợp vào model, Explainable AI (SHAP), triển khai trên Streamlit Web App.

---

## Phạm vi

| Thành phần | Chi tiết |
|------------|----------|
| **Cổ phiếu** | VNM, VCB, FPT, VIC, HPG (5 mã) |
| **Dữ liệu** | Lịch sử 5 năm (Yahoo Finance API qua yfinance) |
| **Input** | Open, High, Low, Close, Volume + 11 Technical Indicators + 3 Sentiment Features |
| **Output** | Giá Close ngày tiếp theo |
| **Features** | 19 features/ngày (khi bật Sentiment) |

---

## Models (5 models)

| Model | Loại | Mô tả |
|-------|------|-------|
| **LSTM** | Deep Learning | Chuỗi 60 ngày × 19 features, ghi nhớ dependency dài hạn |
| **GRU** | Deep Learning | Nhẹ hơn LSTM, ít tham số hơn |
| **XGBoost** | ML Ensemble | Gradient Boosting, hỗ trợ SHAP Explainable AI |
| **Prophet** | Statistical | Trend + Seasonality (Meta/Facebook) |
| **ARIMA** | Statistical | Auto-ARIMA, Time Series cổ điển |

---

## Sentiment Analysis (Tích hợp vào Model)

| Thành phần | Chi tiết |
|------------|----------|
| **Market-based** | Tính từ dữ liệu giá (momentum, volatility, volume signal) — có cho toàn bộ lịch sử |
| **News-based** | Crawl tin tức tài chính (CafeF, VnExpress) — bổ sung cho ngày gần đây |
| **Kỹ thuật** | Từ điển sentiment tiếng Việt + market signals |
| **Output** | 3 features: sentiment_score, sentiment_momentum, sentiment_volatility (-1 → +1) |
| **Ứng dụng** | Đưa vào model như features dự đoán (toggle ON/OFF trên sidebar) |

---

## Explainable AI — SHAP

| Thành phần | Chi tiết |
|------------|----------|
| **Kỹ thuật** | SHAP (SHapley Additive exPlanations) — lý thuyết Shapley (Game Theory) |
| **Áp dụng** | XGBoost model |
| **Output** | Summary Plot, Waterfall Plot, Text Explanation |
| **Ý nghĩa** | Giải thích tại sao model dự đoán giá lên/xuống, feature nào quan trọng nhất (bao gồm sentiment) |

---

## Công nghệ

| Thành phần | Công nghệ |
|------------|-----------| 
| Ngôn ngữ | Python 3.x |
| Data | yfinance (Yahoo Finance API) |
| Deep Learning | TensorFlow/Keras (LSTM, GRU) |
| ML | Scikit-learn, XGBoost |
| Statistical | Prophet, pmdarima |
| Explainable AI | SHAP |
| Sentiment | Market-based (từ giá) + News-based (BeautifulSoup crawl) |
| Frontend | Streamlit |
| Visualization | Plotly |

---

## Đánh giá

- **RMSE** — Root Mean Square Error (phạt nặng sai số lớn)
- **MAE** — Mean Absolute Error (sai số trung bình)
- **MAPE** — Mean Absolute Percentage Error (sai số phần trăm)
- **So sánh** — 5 models trên 5 mã cổ phiếu, có/không có Sentiment
- **Data Split** — Train (80%, trong đó 10% cuối = Validation) / Test (20%) — tránh data leakage

---

## Kết quả mong đợi

1. So sánh 5 models trên 5 mã cổ phiếu
2. Đánh giá ảnh hưởng của Sentiment đến accuracy (có vs không có)
3. Xác định model tốt nhất cho từng loại cổ phiếu
4. SHAP giải thích feature importance (bao gồm sentiment impact)
5. Web app demo tương tác đầy đủ với dashboard

---

## Cấu trúc thư mục

```
stock_prediction/
├── config.py                   # Hằng số cấu hình
├── app.py                      # Streamlit app (6 tabs)
├── requirements.txt            # Dependencies
├── README.md                   # Giới thiệu project
├── SETUP.md                    # Hướng dẫn cài đặt
├── data/                       # Dữ liệu cổ phiếu (auto-generated)
├── saved_models/               # Models đã train (auto-generated)
└── src/
    ├── __init__.py
    ├── data_loader.py          # Tải data từ Yahoo Finance
    ├── preprocessing.py        # Tiền xử lý + Technical Indicators
    ├── evaluation.py           # Đánh giá & so sánh
    ├── explainability.py       # SHAP (Explainable AI)
    ├── sentiment.py            # Market-based + News-based Sentiment
    └── models/
        ├── __init__.py
        ├── lstm_model.py       # LSTM
        ├── gru_model.py        # GRU
        ├── xgboost_model.py    # XGBoost
        ├── prophet_model.py    # Prophet
        └── arima_model.py      # ARIMA
```
