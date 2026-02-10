"""
Cấu hình chung cho project Dự đoán Giá Cổ phiếu Việt Nam.
"""
import os

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# === Danh sách cổ phiếu ===
STOCK_SYMBOLS = ["VNM", "VCB", "FPT", "VIC", "HPG"]

# === Khoảng thời gian lấy dữ liệu ===
DATA_START_DATE = "2021-01-01"
DATA_END_DATE = "2026-02-10"

# === Preprocessing ===
SEQUENCE_LENGTH = 60          # Số ngày lookback cho LSTM/GRU
TEST_RATIO = 0.2              # Tỉ lệ test set
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "close"

# === LSTM / GRU ===
DL_EPOCHS = 50
DL_BATCH_SIZE = 32
DL_LEARNING_RATE = 0.001

# === XGBoost ===
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

# === Prophet ===
PROPHET_CHANGEPOINT_PRIOR = 0.05

# === Tạo thư mục nếu chưa có ===
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
