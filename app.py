"""
📈 Dự đoán Giá Cổ phiếu Việt Nam — Streamlit Web App
Tích hợp 5 models: LSTM, GRU, XGBoost, Prophet, ARIMA
+ Dự đoán tương lai + SHAP Explainability
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STOCK_SYMBOLS, SEQUENCE_LENGTH, DATA_DIR

from src.data_loader import load_stock_data as _load_stock_data, download_stock_data, save_stock_data
from src.preprocessing import (
    add_technical_indicators,
    prepare_data_dl, prepare_data_ml,
    prepare_data_prophet, prepare_data_arima,
    inverse_transform_predictions
)
from src.evaluation import (
    calculate_metrics, compare_models,
    plot_predictions, plot_model_comparison,
    plot_candlestick, plot_training_history
)


@st.cache_data(ttl=300)  # Cache 5 phút, tránh load CSV mỗi lần rerun
def load_stock_data(symbol: str):
    return _load_stock_data(symbol)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="📈 Stock Prediction VN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #4FC3F7;
    }
    .metric-label {
        font-size: 14px;
        color: #90A4AE;
        margin-top: 5px;
    }
    .header-gradient {
        background: linear-gradient(90deg, #4FC3F7, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0d0d2b 100%);
    }
    .prediction-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(33, 150, 243, 0.15));
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-price {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .prediction-label {
        font-size: 16px;
        color: #B0BEC5;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")

    # Chọn mã cổ phiếu
    selected_symbol = st.selectbox(
        "🏢 Mã cổ phiếu",
        STOCK_SYMBOLS,
        index=0
    )

    # Chọn model
    model_options = ["LSTM", "GRU", "XGBoost", "Prophet", "ARIMA"]
    selected_model = st.selectbox(
        "🤖 Model dự đoán",
        model_options,
        index=0
    )

    st.markdown("---")

    # Tham số
    st.markdown("### 🔧 Tham số")
    seq_length = st.slider("Lookback (ngày)", 20, 120, SEQUENCE_LENGTH)
    test_ratio = st.slider("Tỉ lệ Test (%)", 10, 40, 20) / 100
    epochs = st.slider("Epochs (DL)", 10, 100, 50)

    st.markdown("---")

    # Nút tải dữ liệu
    if st.button("📥 Tải/Cập nhật dữ liệu", use_container_width=True):
        with st.spinner(f"Đang tải dữ liệu {selected_symbol}..."):
            df = download_stock_data(selected_symbol)
            if not df.empty:
                save_stock_data(df, selected_symbol)
                st.success(f"✅ Đã tải {len(df)} dòng cho {selected_symbol}")
            else:
                st.error(f"❌ Không thể tải dữ liệu {selected_symbol}")

    st.markdown("---")
    st.markdown("### 📋 Thông tin")
    st.info(
        "**Môn:** Lập trình AI\n\n"
        "**Đề tài:** Dự đoán giá cổ phiếu VN\n\n"
        "**Models:** LSTM, GRU, XGBoost, Prophet, ARIMA"
    )


# ============================================================
# Header
# ============================================================
st.markdown('<p class="header-gradient">📈 Dự đoán Giá Cổ phiếu Việt Nam</p>', unsafe_allow_html=True)
st.markdown(f"**Mã cổ phiếu:** `{selected_symbol}` | **Model:** `{selected_model}` | **Lookback:** `{seq_length}` ngày")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dữ liệu & Biểu đồ",
    "🤖 Đánh giá Model",
    "🔮 Dự đoán Tương lai",
    "📈 So sánh Models",
    "🔍 SHAP - Giải thích AI",
    "📰 Sentiment Analysis"
])


# ============================================================
# Tab 1: Dữ liệu & Biểu đồ
# ============================================================
with tab1:
    st.markdown("### 📊 Dữ liệu Cổ phiếu")

    df = load_stock_data(selected_symbol)

    if df.empty:
        st.warning(f"⚠️ Chưa có dữ liệu cho {selected_symbol}. Hãy bấm '📥 Tải/Cập nhật dữ liệu' ở sidebar.")
    else:
        # Metrics tổng quan
        col1, col2, col3, col4, col5 = st.columns(5)

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = latest['close'] - prev['close']
        change_pct = (change / prev['close']) * 100

        with col1:
            st.metric("💰 Giá Close", f"{latest['close']:,.0f}", f"{change:+,.0f}")
        with col2:
            st.metric("📈 High", f"{latest['high']:,.0f}")
        with col3:
            st.metric("📉 Low", f"{latest['low']:,.0f}")
        with col4:
            st.metric("📊 Volume", f"{latest['volume']:,.0f}")
        with col5:
            st.metric("📅 Số ngày", f"{len(df):,}")

        # Biểu đồ nến
        st.plotly_chart(plot_candlestick(df, selected_symbol), use_container_width=True)

        # Technical Indicators
        with st.expander("📐 Technical Indicators"):
            df_tech = add_technical_indicators(df)
            st.dataframe(df_tech.tail(20), use_container_width=True)

        # Bảng dữ liệu thô
        with st.expander("📋 Dữ liệu thô"):
            st.dataframe(df.tail(50), use_container_width=True)


# ============================================================
# Tab 2: Đánh giá Model (Evaluation)
# ============================================================
with tab2:
    st.markdown(f"### 🤖 Đánh giá {selected_model} trên dữ liệu Test")
    st.info(
        "💡 **Cách đọc kết quả:** Model được train trên 80% dữ liệu lịch sử, "
        "sau đó dự đoán 20% còn lại (data model chưa từng thấy). "
        "Biểu đồ so sánh giá **thực tế** vs **dự đoán** — hai đường càng sát = model càng chính xác."
    )

    df = load_stock_data(selected_symbol)

    if df.empty:
        st.warning("⚠️ Chưa có dữ liệu. Hãy tải dữ liệu trước.")
    else:
        if st.button(f"🚀 Đánh giá {selected_model}", use_container_width=True, type="primary"):
            df_tech = add_technical_indicators(df)

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # ==================== LSTM / GRU ====================
                if selected_model in ["LSTM", "GRU"]:
                    status_text.text("📊 Chuẩn bị dữ liệu...")
                    progress_bar.progress(10)

                    X_train, X_test, y_train, y_test, scaler = prepare_data_dl(
                        df_tech, seq_length=seq_length, test_ratio=test_ratio
                    )
                    n_features = X_train.shape[2]

                    status_text.text(f"🏗️ Xây dựng {selected_model}...")
                    progress_bar.progress(20)

                    if selected_model == "LSTM":
                        from src.models.lstm_model import LSTMModel
                        model = LSTMModel(seq_length, n_features, epochs=epochs)
                    else:
                        from src.models.gru_model import GRUModel
                        model = GRUModel(seq_length, n_features, epochs=epochs)

                    status_text.text(f"🏋️ Training {selected_model}...")
                    progress_bar.progress(30)

                    # Tách 10% cuối của train làm validation (tránh data leakage)
                    val_split = int(len(X_train) * 0.9)
                    X_val = X_train[val_split:]
                    y_val = y_train[val_split:]
                    X_train_actual = X_train[:val_split]
                    y_train_actual = y_train[:val_split]

                    history = model.train(X_train_actual, y_train_actual, X_val, y_val, verbose=0)

                    status_text.text("🔮 Dự đoán...")
                    progress_bar.progress(80)

                    y_pred_scaled = model.predict(X_test)

                    # Inverse transform
                    y_true_real = inverse_transform_predictions(y_test, scaler, n_features)
                    y_pred_real = inverse_transform_predictions(y_pred_scaled, scaler, n_features)

                    # Training history
                    st.plotly_chart(plot_training_history(history), use_container_width=True)

                # ==================== XGBoost ====================
                elif selected_model == "XGBoost":
                    status_text.text("📊 Chuẩn bị dữ liệu...")
                    progress_bar.progress(10)

                    X_train, X_test, y_train, y_test, feature_names = prepare_data_ml(
                        df_tech, test_ratio=test_ratio
                    )

                    status_text.text("🏋️ Training XGBoost...")
                    progress_bar.progress(30)

                    from src.models.xgboost_model import XGBoostModel
                    model = XGBoostModel()
                    model.train(X_train, y_train, X_test, y_test,
                                feature_names=feature_names, verbose=False)

                    status_text.text("🔮 Dự đoán...")
                    progress_bar.progress(80)

                    y_pred_real = model.predict(X_test)
                    y_true_real = y_test

                    # Feature Importance
                    importance = model.get_feature_importance()
                    imp_df = pd.DataFrame.from_dict(importance, orient='index', columns=['Importance'])
                    imp_df = imp_df.sort_values('Importance', ascending=True)

                    fig_imp = go.Figure(go.Bar(
                        x=imp_df['Importance'],
                        y=imp_df.index,
                        orientation='h',
                        marker_color='#4FC3F7'
                    ))
                    fig_imp.update_layout(
                        title='📊 Feature Importance (XGBoost built-in)',
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                # ==================== Prophet ====================
                elif selected_model == "Prophet":
                    status_text.text("📊 Chuẩn bị dữ liệu...")
                    progress_bar.progress(10)

                    train_df, test_df = prepare_data_prophet(df_tech, test_ratio=test_ratio)

                    status_text.text("🏋️ Training Prophet...")
                    progress_bar.progress(30)

                    from src.models.prophet_model import ProphetModel
                    model = ProphetModel()
                    model.train(train_df, verbose=False)

                    status_text.text("🔮 Dự đoán...")
                    progress_bar.progress(80)

                    y_pred_real = model.get_predictions_array(test_df)
                    y_true_real = test_df['y'].values

                # ==================== ARIMA ====================
                elif selected_model == "ARIMA":
                    status_text.text("📊 Chuẩn bị dữ liệu...")
                    progress_bar.progress(10)

                    train_series, test_series = prepare_data_arima(df_tech, test_ratio=test_ratio)

                    status_text.text("🏋️ Training ARIMA (tìm tham số tối ưu)...")
                    progress_bar.progress(30)

                    from src.models.arima_model import ARIMAModel
                    model = ARIMAModel()
                    model.train(train_series, verbose=False)

                    status_text.text("🔮 Dự đoán...")
                    progress_bar.progress(80)

                    y_pred_real = model.predict(n_periods=len(test_series))
                    y_true_real = test_series.values

                # ==================== Kết quả ====================
                progress_bar.progress(100)
                status_text.text("✅ Hoàn thành!")

                # Metrics
                metrics = calculate_metrics(y_true_real, y_pred_real)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['RMSE']:,.2f}</div>
                        <div class="metric-label">RMSE</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['MAE']:,.2f}</div>
                        <div class="metric-label">MAE</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{metrics['MAPE (%)']:.2f}%</div>
                        <div class="metric-label">MAPE</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                # Giải thích metrics
                with st.expander("❓ Các metrics nghĩa là gì?"):
                    st.markdown("""
                    - **RMSE** (Root Mean Square Error): Sai số trung bình. Càng **thấp** càng tốt.
                    - **MAE** (Mean Absolute Error): Trung bình sai lệch tuyệt đối. Ví dụ MAE = 500 nghĩa là model sai trung bình 500 VNĐ.
                    - **MAPE** (Mean Absolute % Error): Sai số theo phần trăm. MAPE < 5% = **rất tốt**, < 10% = **tốt**.
                    """)

                # Biểu đồ predictions
                fig = plot_predictions(
                    y_true_real, y_pred_real,
                    title=f"{selected_model} — {selected_symbol}: Actual vs Predicted (trên Test Set)"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption("📌 *Biểu đồ trên cho thấy model dự đoán trên phần test (20% data cuối) — "
                           "không phải dự đoán tương lai. Để dự đoán tương lai, hãy dùng tab '🔮 Dự đoán Tương lai'.*")

            except Exception as e:
                st.error(f"❌ Lỗi khi chạy {selected_model}: {str(e)}")
                st.exception(e)


# ============================================================
# Tab 3: Dự đoán Tương lai
# ============================================================
with tab3:
    st.markdown("### 🔮 Dự đoán Giá Tương lai")
    st.info(
        "💡 **Cách hoạt động:** Model được train trên **toàn bộ** dữ liệu lịch sử, "
        "sau đó dự đoán giá cho N ngày tiếp theo trong tương lai."
    )

    df = load_stock_data(selected_symbol)

    if df.empty:
        st.warning("⚠️ Chưa có dữ liệu.")
    else:
        future_days = st.slider("📅 Số ngày dự đoán", 1, 30, 7, key="future_days")

        if st.button(f"🔮 Dự đoán {future_days} ngày tới bằng {selected_model}",
                     use_container_width=True, type="primary"):
            df_tech = add_technical_indicators(df)

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                future_predictions = []
                last_date = df['time'].iloc[-1]
                current_close = df['close'].iloc[-1]

                # ==================== LSTM / GRU ====================
                if selected_model in ["LSTM", "GRU"]:
                    status_text.text(f"🏋️ Training {selected_model} trên toàn bộ data...")
                    progress_bar.progress(20)

                    X_all, _, y_all, _, scaler = prepare_data_dl(
                        df_tech, seq_length=seq_length, test_ratio=0.01
                    )
                    n_features = X_all.shape[2]

                    if selected_model == "LSTM":
                        from src.models.lstm_model import LSTMModel
                        model = LSTMModel(seq_length, n_features, epochs=epochs)
                    else:
                        from src.models.gru_model import GRUModel
                        model = GRUModel(seq_length, n_features, epochs=epochs)

                    model.train(X_all, y_all, verbose=0)

                    status_text.text("🔮 Dự đoán tương lai...")
                    progress_bar.progress(60)

                    # Dự đoán lần lượt từng ngày
                    last_sequence = X_all[-1:].copy()
                    for i in range(future_days):
                        pred = model.predict(last_sequence)[0]
                        future_predictions.append(pred)
                        # Shift sequence: bỏ ngày đầu, thêm prediction vào cuối
                        new_step = last_sequence[0, -1, :].copy()
                        new_step[0] = pred  # close ở vị trí 0
                        last_sequence = np.roll(last_sequence, -1, axis=1)
                        last_sequence[0, -1, :] = new_step

                    # Inverse transform
                    future_predictions = inverse_transform_predictions(
                        np.array(future_predictions), scaler, n_features
                    )

                # ==================== XGBoost ====================
                elif selected_model == "XGBoost":
                    status_text.text("🏋️ Training XGBoost trên toàn bộ data...")
                    progress_bar.progress(20)

                    X_all, _, y_all, _, feature_names = prepare_data_ml(
                        df_tech, test_ratio=0.01
                    )

                    from src.models.xgboost_model import XGBoostModel
                    model = XGBoostModel()
                    model.train(X_all, y_all, feature_names=feature_names, verbose=False)

                    status_text.text("🔮 Dự đoán tương lai...")
                    progress_bar.progress(60)

                    # Dự đoán lần lượt
                    last_features = X_all[-1:].copy()
                    close_idx = feature_names.index('close') if 'close' in feature_names else 3
                    for i in range(future_days):
                        pred = model.predict(last_features)[0]
                        future_predictions.append(pred)
                        # Cập nhật close + open/high/low xấp xỉ
                        last_features[0, close_idx] = pred
                        if 'open' in feature_names:
                            last_features[0, feature_names.index('open')] = pred
                        if 'high' in feature_names:
                            last_features[0, feature_names.index('high')] = pred * 1.005
                        if 'low' in feature_names:
                            last_features[0, feature_names.index('low')] = pred * 0.995

                # ==================== Prophet ====================
                elif selected_model == "Prophet":
                    status_text.text("🏋️ Training Prophet trên toàn bộ data...")
                    progress_bar.progress(20)

                    train_df, _ = prepare_data_prophet(df_tech, test_ratio=0.01)

                    from src.models.prophet_model import ProphetModel
                    model = ProphetModel()
                    model.train(train_df, verbose=False)

                    status_text.text("🔮 Dự đoán tương lai...")
                    progress_bar.progress(60)

                    forecast = model.predict(periods=future_days)
                    future_predictions = forecast['yhat'].tail(future_days).values

                # ==================== ARIMA ====================
                elif selected_model == "ARIMA":
                    status_text.text("🏋️ Training ARIMA trên toàn bộ data...")
                    progress_bar.progress(20)

                    train_s, _ = prepare_data_arima(df_tech, test_ratio=0.01)

                    from src.models.arima_model import ARIMAModel
                    model = ARIMAModel()
                    model.train(train_s, verbose=False)

                    status_text.text("🔮 Dự đoán tương lai...")
                    progress_bar.progress(60)

                    future_predictions = model.predict(n_periods=future_days)

                # ==================== Hiển thị kết quả ====================
                progress_bar.progress(100)
                status_text.text("✅ Hoàn thành!")

                future_predictions = np.array(future_predictions).flatten()

                # Tạo ngày tương lai (bỏ T7, CN)
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                              periods=future_days)

                # === Dự đoán ngày mai ===
                tomorrow_pred = future_predictions[0]
                change_from_today = tomorrow_pred - current_close
                change_pct = (change_from_today / current_close) * 100
                direction = "📈" if change_from_today > 0 else "📉"
                color = "#4CAF50" if change_from_today > 0 else "#EF5350"

                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">🔮 Dự đoán giá {selected_symbol} ngày tiếp theo</div>
                    <div class="prediction-price" style="color: {color};">
                        {tomorrow_pred:,.0f} VNĐ
                    </div>
                    <div style="font-size: 18px; color: {color}; margin-top: 8px;">
                        {direction} {change_from_today:+,.0f} VNĐ ({change_pct:+.2f}%)
                    </div>
                    <div style="font-size: 13px; color: #78909C; margin-top: 8px;">
                        Giá hiện tại: {current_close:,.0f} VNĐ | Model: {selected_model}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # === Bảng dự đoán chi tiết ===
                pred_df = pd.DataFrame({
                    'Ngày': future_dates,
                    'Giá dự đoán (VNĐ)': [f"{p:,.0f}" for p in future_predictions],
                    'Thay đổi so với hôm nay': [f"{p - current_close:+,.0f}" for p in future_predictions],
                    'Thay đổi (%)': [f"{((p - current_close) / current_close) * 100:+.2f}%" for p in future_predictions]
                })
                st.dataframe(pred_df, use_container_width=True)

                # === Biểu đồ ===
                # Lấy 30 ngày gần nhất + future
                recent_days = min(60, len(df))
                recent_df = df.tail(recent_days)

                fig = go.Figure()

                # Giá lịch sử
                fig.add_trace(go.Scatter(
                    x=recent_df['time'],
                    y=recent_df['close'],
                    mode='lines',
                    name='Giá lịch sử',
                    line=dict(color='#2196F3', width=2)
                ))

                # Đường nối từ hôm nay đến dự đoán
                bridge_dates = [recent_df['time'].iloc[-1]] + list(future_dates)
                bridge_prices = [current_close] + list(future_predictions)

                fig.add_trace(go.Scatter(
                    x=bridge_dates,
                    y=bridge_prices,
                    mode='lines+markers',
                    name=f'Dự đoán ({selected_model})',
                    line=dict(color='#FF9800', width=2, dash='dash'),
                    marker=dict(size=8, color='#FF9800')
                ))

                # Vùng dự đoán (shading)
                fig.add_vrect(
                    x0=future_dates[0], x1=future_dates[-1],
                    fillcolor="rgba(255, 152, 0, 0.05)",
                    line_width=0,
                    annotation_text="Vùng dự đoán",
                    annotation_position="top left"
                )

                fig.update_layout(
                    title=f'🔮 {selected_symbol} — Dự đoán {future_days} ngày tới ({selected_model})',
                    xaxis_title='Thời gian',
                    yaxis_title='Giá (VNĐ)',
                    template='plotly_dark',
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Disclaimer
                st.warning(
                    "⚠️ **Lưu ý:** Dự đoán giá cổ phiếu chỉ mang tính tham khảo. "
                    "Thị trường chứng khoán bị ảnh hưởng bởi nhiều yếu tố không thể dự đoán được. "
                    "Không nên dùng kết quả này để đưa ra quyết định đầu tư thực tế."
                )
                if future_days > 7:
                    st.info(
                        "📉 **Về độ tin cậy:** Dự đoán càng xa (> 7 ngày) thì sai số tích lũy càng lớn, "
                        "vì mỗi ngày model dùng chính prediction ngày trước làm input (recursive forecasting). "
                        "Nên tập trung vào 3-5 ngày đầu tiên."
                    )

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.exception(e)


# ============================================================
# Tab 4: So sánh Models
# ============================================================
with tab4:
    st.markdown("### 📈 So sánh tất cả Models")

    df = load_stock_data(selected_symbol)

    if df.empty:
        st.warning("⚠️ Chưa có dữ liệu.")
    else:
        models_to_compare = st.multiselect(
            "Chọn models để so sánh:",
            model_options,
            default=model_options
        )

        if st.button("🏁 Chạy so sánh tất cả", use_container_width=True, type="primary"):
            df_tech = add_technical_indicators(df)
            results = {}

            progress = st.progress(0)
            total = len(models_to_compare)

            for i, model_name in enumerate(models_to_compare):
                st.text(f"⏳ Đang chạy {model_name}... ({i+1}/{total})")

                try:
                    if model_name in ["LSTM", "GRU"]:
                        X_train, X_test, y_train, y_test, scaler = prepare_data_dl(
                            df_tech, seq_length=seq_length, test_ratio=test_ratio
                        )
                        n_features = X_train.shape[2]

                        if model_name == "LSTM":
                            from src.models.lstm_model import LSTMModel
                            model = LSTMModel(seq_length, n_features, epochs=epochs)
                        else:
                            from src.models.gru_model import GRUModel
                            model = GRUModel(seq_length, n_features, epochs=epochs)

                        # Tách validation từ train (tránh data leakage)
                        val_split = int(len(X_train) * 0.9)
                        model.train(X_train[:val_split], y_train[:val_split],
                                    X_train[val_split:], y_train[val_split:], verbose=0)
                        y_pred = model.predict(X_test)

                        y_true_real = inverse_transform_predictions(y_test, scaler, n_features)
                        y_pred_real = inverse_transform_predictions(y_pred, scaler, n_features)

                    elif model_name == "XGBoost":
                        X_train, X_test, y_train, y_test, feat = prepare_data_ml(df_tech, test_ratio=test_ratio)
                        from src.models.xgboost_model import XGBoostModel
                        model = XGBoostModel()
                        model.train(X_train, y_train, verbose=False)
                        y_pred_real = model.predict(X_test)
                        y_true_real = y_test

                    elif model_name == "Prophet":
                        train_df, test_df = prepare_data_prophet(df_tech, test_ratio=test_ratio)
                        from src.models.prophet_model import ProphetModel
                        model = ProphetModel()
                        model.train(train_df, verbose=False)
                        y_pred_real = model.get_predictions_array(test_df)
                        y_true_real = test_df['y'].values

                    elif model_name == "ARIMA":
                        train_s, test_s = prepare_data_arima(df_tech, test_ratio=test_ratio)
                        from src.models.arima_model import ARIMAModel
                        model = ARIMAModel()
                        model.train(train_s, verbose=False)
                        y_pred_real = model.predict(n_periods=len(test_s))
                        y_true_real = test_s.values

                    results[model_name] = {
                        'y_true': y_true_real,
                        'y_pred': y_pred_real
                    }

                except Exception as e:
                    st.warning(f"⚠️ {model_name} lỗi: {e}")

                progress.progress((i + 1) / total)

            if results:
                # Bảng so sánh
                comparison_df = compare_models(results)
                st.markdown("#### 📊 Bảng so sánh Metrics")
                st.dataframe(
                    comparison_df.style.highlight_min(
                        subset=['RMSE', 'MAE', 'MAPE (%)'],
                        color='#1B5E20'
                    ),
                    use_container_width=True
                )

                # Biểu đồ so sánh
                st.plotly_chart(
                    plot_model_comparison(comparison_df),
                    use_container_width=True
                )

                # Biểu đồ predictions chồng nhau
                fig = go.Figure()
                colors = {'LSTM': '#2196F3', 'GRU': '#4CAF50', 'XGBoost': '#FF9800',
                          'Prophet': '#9C27B0', 'ARIMA': '#F44336'}

                # Actual line
                first_key = list(results.keys())[0]
                fig.add_trace(go.Scatter(
                    y=results[first_key]['y_true'],
                    mode='lines', name='Actual',
                    line=dict(color='white', width=2)
                ))

                for name, data in results.items():
                    fig.add_trace(go.Scatter(
                        y=data['y_pred'],
                        mode='lines', name=name,
                        line=dict(color=colors.get(name, '#999'), width=1.5, dash='dash')
                    ))

                fig.update_layout(
                    title=f'🔮 So sánh dự đoán — {selected_symbol}',
                    template='plotly_dark',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

                # Best model
                best = comparison_df.iloc[0]
                st.success(
                    f"🏆 **Model tốt nhất:** {best['Model']} "
                    f"(RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f}, MAPE={best['MAPE (%)']:.2f}%)"
                )


# ============================================================
# Tab 5: SHAP — Giải thích AI
# ============================================================
with tab5:
    st.markdown("### 🔍 SHAP — Giải thích tại sao AI dự đoán như vậy")
    st.info(
        "💡 **SHAP (SHapley Additive exPlanations)** cho thấy từng feature (RSI, MACD, Volume...) "
        "đóng góp bao nhiêu vào kết quả dự đoán. Giúp hiểu **tại sao** model quyết định giá lên hay xuống.\n\n"
        "⚡ SHAP chỉ hoạt động với **XGBoost** (model ML tabular)."
    )

    df = load_stock_data(selected_symbol)

    if df.empty:
        st.warning("⚠️ Chưa có dữ liệu.")
    else:
        if st.button("🔬 Phân tích SHAP (XGBoost)", use_container_width=True, type="primary"):
            df_tech = add_technical_indicators(df)

            with st.spinner("Đang train XGBoost & tính SHAP values..."):
                try:
                    from src.explainability import (
                        compute_shap_values, plot_shap_summary,
                        plot_shap_waterfall, get_shap_explanation_text
                    )

                    # Train XGBoost
                    X_train, X_test, y_train, y_test, feature_names = prepare_data_ml(
                        df_tech, test_ratio=test_ratio
                    )

                    from src.models.xgboost_model import XGBoostModel
                    model = XGBoostModel()
                    model.train(X_train, y_train, feature_names=feature_names, verbose=False)

                    y_pred = model.predict(X_test)

                    # Tính SHAP values
                    shap_values, explainer = compute_shap_values(model, X_test, feature_names)

                    # === 1. SHAP Summary Plot ===
                    st.markdown("#### 📊 Feature Importance (SHAP)")
                    st.caption("Feature nào ảnh hưởng nhiều nhất đến dự đoán giá trên toàn bộ test set.")
                    st.plotly_chart(
                        plot_shap_summary(shap_values, X_test, feature_names),
                        use_container_width=True
                    )

                    # === 2. Giải thích prediction gần nhất ===
                    st.markdown("---")
                    st.markdown("#### 🔬 Giải thích dự đoán gần nhất")
                    st.caption("Tại sao model dự đoán giá như vậy cho ngày gần nhất.")

                    last_idx = -1
                    last_pred = y_pred[last_idx]
                    base_value = explainer.expected_value
                    if isinstance(base_value, np.ndarray):
                        base_value = base_value[0]

                    # Text explanation
                    explanation_text = get_shap_explanation_text(
                        shap_values[last_idx], feature_names,
                        X_test[last_idx], last_pred
                    )
                    st.markdown(explanation_text)

                    # Waterfall chart
                    st.plotly_chart(
                        plot_shap_waterfall(
                            shap_values[last_idx], X_test[last_idx],
                            feature_names, base_value, last_pred
                        ),
                        use_container_width=True
                    )

                    # === 3. So sánh SHAP vs Built-in Importance ===
                    st.markdown("---")
                    st.markdown("#### 📊 SHAP vs XGBoost Built-in Feature Importance")
                    st.caption("SHAP chính xác hơn vì dựa trên lý thuyết Shapley (Game Theory).")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**SHAP Importance**")
                        mean_shap = np.abs(shap_values).mean(axis=0)
                        shap_imp = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP': mean_shap
                        }).sort_values('SHAP', ascending=False)
                        st.dataframe(shap_imp, use_container_width=True)

                    with col2:
                        st.markdown("**XGBoost Built-in**")
                        builtin_imp = model.get_feature_importance()
                        builtin_df = pd.DataFrame.from_dict(
                            builtin_imp, orient='index', columns=['Importance']
                        ).sort_values('Importance', ascending=False)
                        builtin_df.index.name = 'Feature'
                        st.dataframe(builtin_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Lỗi SHAP: {str(e)}")
                    st.exception(e)


# ============================================================
# Tab 6: Sentiment Analysis
# ============================================================
with tab6:
    st.markdown("### 📰 Sentiment Analysis")
    st.info("Phân tích cảm xúc từ tin tức tài chính liên quan đến mã cổ phiếu.")

    if st.button(f"🔍 Phân tích Sentiment cho {selected_symbol}", use_container_width=True):
        with st.spinner("Đang crawl & phân tích tin tức..."):
            try:
                from src.sentiment import get_sentiment_for_stock, save_sentiment_data

                sentiment_df = get_sentiment_for_stock(selected_symbol)

                if sentiment_df.empty:
                    st.warning("Không tìm được tin tức. Có thể do giới hạn truy cập.")
                else:
                    save_sentiment_data(sentiment_df, selected_symbol)

                    # Thống kê
                    avg_score = sentiment_df['sentiment_score'].mean()
                    pos_count = (sentiment_df['sentiment_score'] > 0).sum()
                    neg_count = (sentiment_df['sentiment_score'] < 0).sum()
                    neu_count = (sentiment_df['sentiment_score'] == 0).sum()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        emoji = "📈" if avg_score > 0 else "📉" if avg_score < 0 else "➡️"
                        st.metric(f"{emoji} Trung bình", f"{avg_score:.3f}")
                    with col2:
                        st.metric("😊 Tích cực", pos_count)
                    with col3:
                        st.metric("😞 Tiêu cực", neg_count)
                    with col4:
                        st.metric("😐 Trung lập", neu_count)

                    # Biểu đồ phân bố
                    fig_sent = go.Figure()
                    fig_sent.add_trace(go.Histogram(
                        x=sentiment_df['sentiment_score'],
                        nbinsx=20,
                        marker_color='#4FC3F7',
                        name='Sentiment Score'
                    ))
                    fig_sent.update_layout(
                        title='📊 Phân bố Sentiment Score',
                        xaxis_title='Score (-1 = Tiêu cực, +1 = Tích cực)',
                        yaxis_title='Số lượng',
                        template='plotly_dark',
                        height=400
                    )
                    st.plotly_chart(fig_sent, use_container_width=True)

                    # Bảng tin tức
                    st.markdown("#### 📋 Danh sách tin tức")
                    st.dataframe(
                        sentiment_df.sort_values('sentiment_score', ascending=False),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
                st.exception(e)

    # Hiển thị data đã lưu (nếu có)
    sentiment_file = os.path.join(DATA_DIR, f"{selected_symbol}_sentiment.csv")
    if os.path.exists(sentiment_file):
        with st.expander("📂 Dữ liệu Sentiment đã lưu"):
            saved_df = pd.read_csv(sentiment_file)
            st.dataframe(saved_df, use_container_width=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #90A4AE;'>"
    "📈 Stock Prediction VN | Lập trình Trí tuệ Nhân tạo | "
    "LSTM • GRU • XGBoost • Prophet • ARIMA • SHAP"
    "</div>",
    unsafe_allow_html=True
)
