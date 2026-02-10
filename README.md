# 📈 Dự đoán Giá Cổ phiếu Việt Nam

> **Môn học:** Lập trình Trí tuệ Nhân tạo  
> Dự đoán giá cổ phiếu VN sử dụng 5 mô hình ML/DL + Explainable AI (SHAP) + Sentiment Analysis  
> Giao diện web tương tác bằng Streamlit

---

## 🎯 Tổng quan

Hệ thống dự đoán giá cổ phiếu Việt Nam cho **5 mã**: VNM, VCB, FPT, VIC, HPG.

### Các tính năng chính

- **5 mô hình dự đoán** — LSTM, GRU, XGBoost, Prophet, ARIMA
- **Đánh giá model** — Train 80% / Test 20%, so sánh Actual vs Predicted
- **Dự đoán tương lai** — Dự đoán giá 1-30 ngày tiếp theo (chưa xảy ra)
- **So sánh models** — Chạy đồng thời 5 models, xếp hạng theo RMSE/MAE/MAPE
- **Explainable AI (SHAP)** — Giải thích tại sao XGBoost dự đoán giá như vậy
- **Sentiment Analysis** — Phân tích cảm xúc tin tức tài chính từ CafeF/VnExpress
- **Giao diện Streamlit** — 6 tabs, dark theme, biểu đồ tương tác Plotly

---

## 🏗️ Kiến trúc hệ thống

```
Yahoo Finance API ──→ Data (CSV) ──→ Preprocessing ──→ Models ──→ Prediction
                                        │                           │
                                   Technical               ┌───────┴────────┐
                                   Indicators               │                │
                                   (16 features)        Đánh giá          Dự đoán
                                                      (80/20 split)     (Tương lai)
                                                           │                │
                                                     RMSE/MAE/MAPE    Giá N ngày tới
                                                           │
                                                      SHAP (XGBoost)
                                                           │
                                                   Giải thích prediction

CafeF / VnExpress ──→ Crawl Headlines ──→ Sentiment Score ──→ Hiển thị (chưa tích hợp vào model)
```

---

## 🤖 Mô hình dự đoán

| Model | Loại | Input | Cách hoạt động |
|-------|------|-------|----------------|
| **LSTM** | Deep Learning | Chuỗi 60 ngày × 16 features (3D) | Long Short-Term Memory — ghi nhớ pattern dài hạn qua cổng quên/nhớ |
| **GRU** | Deep Learning | Chuỗi 60 ngày × 16 features (3D) | Gated Recurrent Unit — đơn giản hơn LSTM, ít tham số hơn |
| **XGBoost** | Machine Learning | 16 features (bảng 2D) | Gradient Boosting — ensemble nhiều decision trees, hỗ trợ SHAP |
| **Prophet** | Statistical | 2 cột: ngày + giá Close | Phân tách trend + seasonality (Meta/Facebook) |
| **ARIMA** | Statistical | 1 cột: giá Close | Mô hình tự hồi quy chuỗi thời gian cổ điển |

### Cách đánh giá (Tab "Đánh giá Model")

```
Dữ liệu N ngày
├── 80% đầu → TRAIN: model học patterns
└── 20% cuối → TEST: model dự đoán, so sánh với giá thực tế
    → Output: biểu đồ Actual vs Predicted + RMSE / MAE / MAPE
```

### Dự đoán tương lai (Tab "Dự đoán Tương lai")

```
Dữ liệu N ngày
└── 100% → TRAIN trên toàn bộ data
    └── Dự đoán M ngày tiếp theo (chưa xảy ra)
        → Output: bảng giá + biểu đồ kết nối lịch sử → tương lai
```

---

## 🔧 Tiền xử lý dữ liệu

Từ 5 cột gốc (OHLCV), tạo thêm **11 technical indicators**:

| Feature | Ý nghĩa |
|---------|---------|
| SMA 10 / 20 / 50 | Trung bình trượt đơn giản (ngắn / trung / dài hạn) |
| EMA 12 / 26 | Trung bình hàm mũ (phản ứng nhanh hơn SMA) |
| RSI 14 | Relative Strength Index — quá mua (>70) / quá bán (<30) |
| MACD | Moving Average Convergence Divergence — động lượng |
| MACD Signal | Đường tín hiệu MACD |
| Price Change | Biến động giá tuyệt đối |
| Price Change % | Biến động giá phần trăm |
| Volume Change % | Biến động khối lượng |

→ Tổng: **16 features** cho mỗi ngày giao dịch, dữ liệu được chuẩn hóa (MinMaxScaler) trước khi đưa vào model.

---

## 🔍 Explainable AI — SHAP

**SHAP (SHapley Additive exPlanations)** giải thích tại sao XGBoost dự đoán giá như vậy, dựa trên lý thuyết Shapley (Game Theory).

| Visualization | Ý nghĩa |
|--------------|---------|
| **Summary Plot** | Feature nào quan trọng nhất trên toàn bộ test set (bar chart) |
| **Waterfall Plot** | Giải thích 1 prediction cụ thể — feature nào đẩy giá lên/xuống bao nhiêu VNĐ |
| **Text Explanation** | "close = 69,100 → đẩy giá LÊN 2,961 VNĐ" |
| **So sánh** | SHAP Importance vs XGBoost Built-in Feature Importance |

> SHAP chỉ áp dụng cho XGBoost (tree-based model). LSTM/GRU cần kỹ thuật XAI khác.

---

## 📰 Sentiment Analysis

Phân tích cảm xúc tin tức tài chính liên quan đến cổ phiếu:

- **Crawl tin tức** từ CafeF và VnExpress
- **Phân tích sentiment** bằng từ điển tiếng Việt (score -1 → +1)
- **Hiển thị**: histogram phân bố, danh sách tin + điểm sentiment

> **Lưu ý:** Sentiment hiện tại là module **demo riêng**, chưa tích hợp vào features của model dự đoán.

---

## � Metrics đánh giá

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **RMSE** | √(Σ(ŷ-y)²/n) | Sai số trung bình bình phương — phạt nặng sai số lớn |
| **MAE** | Σ\|ŷ-y\|/n | Sai số trung bình tuyệt đối — dễ hiểu |
| **MAPE** | Σ\|ŷ-y\|/y × 100% | Sai số phần trăm — so sánh được giữa các mã cổ phiếu |

---

## 🖥️ Giao diện Streamlit (6 Tabs)

| Tab | Chức năng |
|-----|-----------|
| 📊 **Dữ liệu & Biểu đồ** | Biểu đồ nến (candlestick), volume, technical indicators, data thô |
| ⚙️ **Đánh giá Model** | Train 80% → Test 20% → biểu đồ Actual vs Predicted + metrics |
| 🔮 **Dự đoán Tương lai** | Dự đoán 1-30 ngày tới, bảng giá chi tiết + biểu đồ forecast |
| 🏆 **So sánh Models** | Chạy cả 5 models → bảng xếp hạng theo RMSE/MAE/MAPE |
| 🔍 **SHAP - Giải thích AI** | SHAP summary, waterfall, text explanation (chỉ XGBoost) |
| 📰 **Sentiment Analysis** | Crawl tin tức → phân tích cảm xúc → histogram + bảng |

---

## �🛠️ Cài đặt & Chạy

### 1. Tạo Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
```

### 2. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 3. Tải dữ liệu
```bash
python src/data_loader.py
```
Hoặc bấm nút **"Tải/Cập nhật dữ liệu"** trên sidebar của app.

### 4. Chạy Web App
```bash
streamlit run app.py
```
Mở trình duyệt tại `http://localhost:8501`

---

## 📁 Cấu trúc Project

```
stock_prediction/
├── config.py                   # Hằng số cấu hình (tham số model, paths)
├── app.py                      # Streamlit Web App (6 tabs)
├── requirements.txt            # Python dependencies
├── README.md                   # File này
├── data/                       # Dữ liệu CSV (auto-generated)
│   ├── VNM.csv
│   ├── VCB.csv
│   ├── FPT.csv
│   ├── VIC.csv
│   └── HPG.csv
├── saved_models/               # Models đã train (auto-generated)
└── src/
    ├── __init__.py
    ├── data_loader.py           # Tải dữ liệu từ Yahoo Finance (yfinance)
    ├── preprocessing.py         # Tiền xử lý + 11 Technical Indicators
    ├── evaluation.py            # Metrics (RMSE/MAE/MAPE) + biểu đồ Plotly
    ├── explainability.py        # SHAP — Explainable AI cho XGBoost
    ├── sentiment.py             # Crawl tin tức + Phân tích cảm xúc VN
    └── models/
        ├── __init__.py
        ├── lstm_model.py        # LSTM (2 layers, Dropout, EarlyStopping)
        ├── gru_model.py         # GRU (2 layers, Dropout, EarlyStopping)
        ├── xgboost_model.py     # XGBoost Regressor
        ├── prophet_model.py     # Prophet (Meta)
        └── arima_model.py       # Auto-ARIMA (pmdarima)
```

---

## � Công nghệ sử dụng

| Thành phần | Công nghệ |
|-----------|-----------|
| Ngôn ngữ | Python 3.x |
| Deep Learning | TensorFlow / Keras (LSTM, GRU) |
| Machine Learning | XGBoost, Scikit-learn |
| Statistical | Prophet (Meta), pmdarima (ARIMA) |
| Explainable AI | SHAP |
| NLP | BeautifulSoup (crawl), Từ điển sentiment VN |
| Data | yfinance (Yahoo Finance API) |
| Web App | Streamlit |
| Visualization | Plotly (interactive charts) |
| Data Processing | Pandas, NumPy |

---

## ⚠️ Lưu ý

- Dự đoán giá cổ phiếu **chỉ mang tính tham khảo**, không nên dùng để đưa ra quyết định đầu tư thực tế.
- Dữ liệu tải về theo **batch** (không phải realtime), phù hợp cho mục đích nghiên cứu và demo.
- Thị trường chứng khoán bị ảnh hưởng bởi nhiều yếu tố không thể dự đoán (chính sách, thiên tai, tâm lý đám đông...).
