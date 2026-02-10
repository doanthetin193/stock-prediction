# 📈 Dự đoán Giá Cổ phiếu Việt Nam

> **Môn học:** Lập trình Trí tuệ Nhân tạo  
> **Yêu cầu:** Đề tài có yếu tố AI (Machine Learning / Deep Learning / các kỹ thuật hiện đại)

---

## Tổng quan
Xây dựng hệ thống dự đoán giá cổ phiếu VN sử dụng nhiều mô hình ML/DL, kết hợp Sentiment Analysis từ tin tức, triển khai trên Streamlit Web App.

---

## Phạm vi

| Thành phần | Chi tiết |
|------------|----------|
| **Cổ phiếu** | VNM, VCB, FPT, VIC, HPG (5 mã) |
| **Dữ liệu** | Lịch sử 3-5 năm (vnstock API) |
| **Input** | Open, High, Low, Close, Volume + Sentiment Score |
| **Output** | Giá Close ngày tiếp theo |

---

## Models (5 models)

| Model | Loại | Mô tả |
|-------|------|-------|
| **LSTM** | Deep Learning | Học dependency dài hạn |
| **GRU** | Deep Learning | Nhẹ hơn LSTM |
| **Prophet** | Statistical | Trend + Seasonality |
| **XGBoost** | ML Ensemble | Gradient Boosting |
| **ARIMA** | Statistical | Time Series cổ điển |

---

## Sentiment Analysis (Bonus)

| Thành phần | Chi tiết |
|------------|----------|
| **Nguồn** | Tin tức tài chính (CafeF, VnExpress) |
| **Kỹ thuật** | NLP - PhoBERT hoặc TextBlob |
| **Output** | Sentiment Score (-1 đến +1) |
| **Ứng dụng** | Thêm vào features để dự đoán |

---

## Công nghệ

| Thành phần | Công nghệ |
|------------|-----------|
| Ngôn ngữ | Python |
| Data | vnstock, BeautifulSoup (crawl) |
| Deep Learning | TensorFlow/Keras |
| ML | Scikit-learn, XGBoost |
| NLP | PhoBERT / TextBlob |
| Statistical | Prophet, statsmodels |
| Frontend | Streamlit |
| Visualization | Plotly |

---

## Đánh giá

- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error  
- **MAPE** - Mean Absolute Percentage Error
- **So sánh** - 5 models trên 5 mã cổ phiếu

---

## Kết quả mong đợi

1. So sánh 5 models trên 5 mã cổ phiếu
2. Đánh giá ảnh hưởng của Sentiment đến giá
3. Xác định model tốt nhất cho từng loại
4. Web app demo tương tác đầy đủ
5. Dashboard với biểu đồ nến, volume, dự đoán

---

## Cấu trúc thư mục (dự kiến)

```
stock_prediction/
├── data/                   # Dữ liệu cổ phiếu + tin tức
├── models/                 # Saved models
├── src/
│   ├── data_loader.py      # Lấy data từ vnstock
│   ├── preprocessing.py    # Tiền xử lý
│   ├── sentiment.py        # Phân tích sentiment tin tức
│   ├── lstm_model.py       # LSTM
│   ├── gru_model.py        # GRU
│   ├── prophet_model.py    # Prophet
│   ├── xgboost_model.py    # XGBoost
│   ├── arima_model.py      # ARIMA
│   └── evaluation.py       # Đánh giá & so sánh
├── app.py                  # Streamlit app
├── requirements.txt
└── README.md
```
