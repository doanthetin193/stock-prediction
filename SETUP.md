# 🚀 Hướng dẫn Cài đặt & Chạy Project

Hướng dẫn từng bước để chạy project **Dự đoán Giá Cổ phiếu Việt Nam** trên máy local.

---

## 📋 Yêu cầu hệ thống

- **Python** 3.9 trở lên ([Tải tại đây](https://www.python.org/downloads/))
- **Git** ([Tải tại đây](https://git-scm.com/downloads))
- **RAM** tối thiểu 4GB (khuyến nghị 8GB cho Deep Learning models)
- **Dung lượng ổ đĩa** ~1GB (bao gồm dependencies)

---

## 📥 Bước 1: Clone project

```bash
git clone https://github.com/doanthetin193/stock-prediction.git
cd stock-prediction
```

Sau khi clone, cấu trúc thư mục sẽ như sau:

```
stock-prediction/
├── config.py                   # Hằng số cấu hình
├── app.py                      # Streamlit Web App (6 tabs)
├── requirements.txt            # Danh sách thư viện cần cài
├── README.md                   # Giới thiệu project
├── SETUP.md                    # File hướng dẫn này
├── .gitignore                  # Danh sách file không push lên GitHub
└── src/                        # Source code chính
    ├── __init__.py
    ├── data_loader.py           # Tải dữ liệu từ Yahoo Finance
    ├── preprocessing.py         # Tiền xử lý + Technical Indicators
    ├── evaluation.py            # Metrics + Biểu đồ
    ├── explainability.py        # SHAP (Explainable AI)
    ├── sentiment.py             # Phân tích cảm xúc tin tức
    └── models/
        ├── __init__.py
        ├── lstm_model.py        # LSTM
        ├── gru_model.py         # GRU
        ├── xgboost_model.py     # XGBoost
        ├── prophet_model.py     # Prophet
        └── arima_model.py       # ARIMA
```

> **Lưu ý:** Thư mục `data/`, `saved_models/`, `.venv/` chưa có — sẽ được tạo ở các bước tiếp theo.

---

## 🐍 Bước 2: Tạo Virtual Environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Khi activate thành công, terminal sẽ hiển thị `(.venv)` ở đầu dòng.

---

## 📦 Bước 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

Quá trình cài đặt sẽ mất **3-5 phút** (TensorFlow, XGBoost, Prophet, SHAP...).

Nếu gặp lỗi, thử:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Bước 4: Tải dữ liệu cổ phiếu

### Cách 1 — Command line (khuyến nghị cho lần đầu)
```bash
python src/data_loader.py
```

### Cách 2 — Trên Web App
Bấm nút **"📥 Tải/Cập nhật dữ liệu"** ở sidebar (sau khi chạy app ở Bước 5).

Sau khi tải xong, thư mục `data/` sẽ xuất hiện:
```
data/
├── VNM.csv      # Vinamilk
├── VCB.csv      # Vietcombank
├── FPT.csv      # FPT Corporation
├── VIC.csv      # Vingroup
└── HPG.csv      # Hòa Phát Group
```

---

## 🌐 Bước 5: Chạy Web App

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại **http://localhost:8501**.

Nếu trình duyệt không tự mở, hãy copy link trên và paste vào trình duyệt.

---

## 🎮 Bước 6: Sử dụng App

### Sidebar (bên trái)
1. **Chọn mã cổ phiếu**: VNM, VCB, FPT, VIC, HPG
2. **Chọn model**: LSTM, GRU, XGBoost, Prophet, ARIMA
3. **Tham số**: Lookback (ngày), Tỉ lệ Test (%), Epochs (DL)

### 6 Tabs

| Tab | Hướng dẫn |
|-----|-----------|
| **📊 Dữ liệu & Biểu đồ** | Xem ngay — biểu đồ nến, volume, technical indicators |
| **⚙️ Đánh giá Model** | Bấm **"Train & Đánh giá"** → chờ train xong → xem kết quả |
| **🔮 Dự đoán Tương lai** | Chọn số ngày → Bấm **"Dự đoán"** → xem giá tương lai |
| **🏆 So sánh Models** | Bấm **"So sánh tất cả"** → chờ 5 models chạy xong → xem xếp hạng |
| **🔍 SHAP - Giải thích AI** | Bấm **"Phân tích SHAP"** → xem tại sao XGBoost dự đoán như vậy |
| **📰 Sentiment Analysis** | Bấm **"Phân tích"** → crawl tin tức → xem cảm xúc thị trường |

> **Lưu ý:** Tab SHAP chỉ hoạt động với XGBoost. Các model khác chưa hỗ trợ SHAP.

---

## ⏱️ Thời gian chạy ước tính

| Thao tác | Thời gian |
|----------|-----------|
| Cài dependencies | 3-5 phút |
| Tải data (5 mã) | 10-30 giây |
| Train LSTM/GRU (50 epochs) | 1-3 phút |
| Train XGBoost | 5-10 giây |
| Train Prophet | 10-30 giây |
| Train ARIMA | 10-30 giây |
| SHAP Analysis | 5-15 giây |

---

## ❓ Xử lý lỗi thường gặp

### Lỗi `ModuleNotFoundError`
```bash
pip install -r requirements.txt    # Cài lại dependencies
```

### Lỗi `No data found` khi chạy app
```bash
python src/data_loader.py          # Tải data trước
```

### Lỗi TensorFlow trên máy yếu
Nếu LSTM/GRU chạy chậm hoặc lỗi, hãy dùng XGBoost/Prophet/ARIMA (nhẹ hơn).

### Lỗi port 8501 đã bị chiếm
```bash
streamlit run app.py --server.port 8502
```

---

## 📝 Tóm tắt nhanh (Quick Start)

```bash
# 1. Clone
git clone https://github.com/doanthetin193/stock-prediction.git
cd stock-prediction

# 2. Setup
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt

# 3. Tải data
python src/data_loader.py

# 4. Chạy app
streamlit run app.py
```
