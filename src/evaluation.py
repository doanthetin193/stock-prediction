"""
Module đánh giá và so sánh performance của các models.
Metrics: RMSE, MAE, MAPE
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính các metrics đánh giá.

    Args:
        y_true: giá trị thực
        y_pred: giá trị dự đoán

    Returns:
        dict chứa RMSE, MAE, MAPE
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Đảm bảo cùng kích thước
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE — tránh chia cho 0
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE (%)': round(mape, 4)
    }


def compare_models(results: dict) -> pd.DataFrame:
    """
    So sánh kết quả của nhiều models.

    Args:
        results: dict {model_name: {'y_true': ..., 'y_pred': ...}}

    Returns:
        DataFrame với các metrics cho từng model
    """
    comparison = []
    for model_name, data in results.items():
        metrics = calculate_metrics(data['y_true'], data['y_pred'])
        metrics['Model'] = model_name
        comparison.append(metrics)

    df = pd.DataFrame(comparison)
    df = df[['Model', 'RMSE', 'MAE', 'MAPE (%)']].sort_values('RMSE')
    df.reset_index(drop=True, inplace=True)

    return df


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     title: str = "Actual vs Predicted",
                     dates=None) -> go.Figure:
    """
    Biểu đồ so sánh giá thực vs dự đoán.

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    x_axis = dates if dates is not None else list(range(len(y_true)))

    fig.add_trace(go.Scatter(
        x=x_axis, y=y_true,
        mode='lines', name='Actual',
        line=dict(color='#2196F3', width=2)
    ))

    min_len = min(len(y_true), len(y_pred))
    x_pred = x_axis[-min_len:] if dates is not None else list(range(len(y_true) - min_len, len(y_true)))

    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred[:min_len],
        mode='lines', name='Predicted',
        line=dict(color='#FF5722', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Price (VNĐ)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99),
        height=500
    )

    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Biểu đồ bar chart so sánh metrics giữa các models.

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['RMSE', 'MAE', 'MAPE (%)']
    )

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
    models = comparison_df['Model'].tolist()

    for i, metric in enumerate(['RMSE', 'MAE', 'MAPE (%)'], 1):
        fig.add_trace(
            go.Bar(
                x=models,
                y=comparison_df[metric],
                marker_color=colors[:len(models)],
                name=metric,
                showlegend=False
            ),
            row=1, col=i
        )

    fig.update_layout(
        title='📊 So sánh Performance giữa các Models',
        template='plotly_dark',
        height=400,
        showlegend=False
    )

    return fig


def plot_training_history(history) -> go.Figure:
    """
    Biểu đồ training loss/val_loss cho DL models.

    Args:
        history: Keras History object

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    epochs = list(range(1, len(history.history['loss']) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=history.history['loss'],
        mode='lines', name='Training Loss',
        line=dict(color='#2196F3', width=2)
    ))

    if 'val_loss' in history.history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history.history['val_loss'],
            mode='lines', name='Validation Loss',
            line=dict(color='#FF5722', width=2)
        ))

    fig.update_layout(
        title='📉 Training & Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss (MSE)',
        template='plotly_dark',
        height=400
    )

    return fig


def plot_candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Biểu đồ nến (candlestick) cho cổ phiếu.

    Args:
        df: DataFrame với cột time, open, high, low, close, volume
        symbol: mã cổ phiếu

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ),
        row=1, col=1
    )

    # Volume bar chart
    colors = ['#26A69A' if c >= o else '#EF5350'
              for c, o in zip(df['close'], df['open'])]

    fig.add_trace(
        go.Bar(
            x=df['time'],
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            showlegend=False
        ),
        row=2, col=1
    )

    fig.update_layout(
        title=f'📈 {symbol} - Biểu đồ Nến & Volume',
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=False
    )

    fig.update_yaxes(title_text='Giá (VNĐ)', row=1, col=1)
    fig.update_yaxes(title_text='Volume', row=2, col=1)

    return fig
