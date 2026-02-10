"""
Module Explainability (XAI) — SHAP cho XGBoost.
Giải thích tại sao model dự đoán giá cổ phiếu như vậy.
"""
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _patch_shap_xgboost():
    """
    Patch SHAP để tương thích với XGBoost >= 3.0.
    XGBoost 3.x lưu base_score dạng '[5.866]' thay vì '5.866',
    gây lỗi ValueError khi SHAP parse bằng float().
    """
    import shap.explainers._tree as shap_tree

    if hasattr(shap_tree, '_original_set_xgb_attrs'):
        return  # Đã patch rồi

    original_fn = shap_tree.TreeEnsemble._set_xgboost_model_attributes

    def patched_fn(self, *args, **kwargs):
        try:
            return original_fn(self, *args, **kwargs)
        except ValueError as e:
            if 'could not convert string to float' in str(e):
                # Lấy model object — đây là booster hoặc XGBRegressor
                model = args[0] if args else self.model
                import json as _json
                if hasattr(model, 'save_config'):
                    config = _json.loads(model.save_config())
                    bs = config['learner']['learner_model_param']['base_score']
                    config['learner']['learner_model_param']['base_score'] = bs.strip('[]')
                    model.load_config(_json.dumps(config))
                    return original_fn(self, *args, **kwargs)
            raise

    shap_tree.TreeEnsemble._set_xgboost_model_attributes = patched_fn
    shap_tree._original_set_xgb_attrs = True


def compute_shap_values(model, X_data: np.ndarray, feature_names: list):
    """
    Tính SHAP values cho XGBoost model.

    Args:
        model: XGBoostModel instance (có attribute .model)
        X_data: dữ liệu cần giải thích (thường là X_test)
        feature_names: tên các features

    Returns:
        shap_values: mảng SHAP values
        explainer: SHAP TreeExplainer
    """
    import json as _json

    xgb_model = model.model if hasattr(model, 'model') else model

    # Patch SHAP cho XGBoost 3.x compatibility
    _patch_shap_xgboost()

    # Lấy base_score trước
    booster = xgb_model.get_booster()
    config = _json.loads(booster.save_config())
    base_score_str = config['learner']['learner_model_param']['base_score']
    base_score = float(base_score_str.strip('[]'))

    try:
        explainer = shap.TreeExplainer(xgb_model)
    except ValueError:
        # Nếu vẫn lỗi, dùng KernelExplainer (chậm hơn nhưng luôn hoạt động)
        import pandas as pd
        X_bg = shap.sample(pd.DataFrame(X_data, columns=feature_names), min(50, len(X_data)))
        explainer = shap.KernelExplainer(xgb_model.predict, X_bg)
        explainer.expected_value = base_score

    shap_values = explainer.shap_values(X_data)

    # Đảm bảo expected_value đúng
    if not hasattr(explainer, 'expected_value') or explainer.expected_value is None:
        explainer.expected_value = base_score

    return shap_values, explainer



def plot_shap_summary(shap_values: np.ndarray, X_data: np.ndarray,
                      feature_names: list) -> go.Figure:
    """
    Biểu đồ SHAP Summary — feature nào quan trọng nhất (tổng quan).

    Returns:
        Plotly Figure
    """
    # Tính mean |SHAP| cho mỗi feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)

    fig = go.Figure(go.Bar(
        x=mean_shap[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=mean_shap[sorted_idx],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='|SHAP|')
        )
    ))

    fig.update_layout(
        title='🔍 SHAP Summary — Feature Importance (Mean |SHAP value|)',
        xaxis_title='Mean |SHAP value| (tác động trung bình đến giá)',
        yaxis_title='Feature',
        template='plotly_dark',
        height=max(400, len(feature_names) * 30)
    )

    return fig


def plot_shap_waterfall(shap_values: np.ndarray, X_single: np.ndarray,
                        feature_names: list, base_value: float,
                        prediction: float) -> go.Figure:
    """
    Biểu đồ SHAP Waterfall — giải thích 1 prediction cụ thể.
    Cho thấy từng feature đẩy giá lên hay kéo giá xuống bao nhiêu.

    Args:
        shap_values: SHAP values cho 1 sample
        X_single: giá trị features của sample đó
        feature_names: tên features
        base_value: giá trị cơ sở (trung bình)
        prediction: giá trị dự đoán cuối cùng

    Returns:
        Plotly Figure
    """
    # Sắp xếp theo |SHAP| giảm dần
    abs_shap = np.abs(shap_values)
    sorted_idx = np.argsort(abs_shap)[::-1]

    # Lấy top 10 features quan trọng nhất
    top_n = min(10, len(feature_names))
    top_idx = sorted_idx[:top_n]

    labels = []
    values = []
    colors = []

    for idx in reversed(top_idx):
        feat_name = feature_names[idx]
        feat_val = X_single[idx]
        shap_val = shap_values[idx]

        labels.append(f"{feat_name} = {feat_val:,.2f}")
        values.append(shap_val)
        colors.append('#EF5350' if shap_val < 0 else '#26A69A')

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+,.1f}" for v in values],
        textposition='outside'
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title=f'🔬 SHAP Waterfall — Giải thích dự đoán: {prediction:,.0f} VNĐ',
        xaxis_title='SHAP value (tác động đến giá)',
        template='plotly_dark',
        height=max(400, top_n * 45),
        annotations=[
            dict(
                text=f"Base value (trung bình): {base_value:,.0f}",
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                showarrow=False,
                font=dict(color='#90A4AE', size=12)
            )
        ]
    )

    return fig


def plot_shap_beeswarm(shap_values: np.ndarray, X_data: np.ndarray,
                        feature_names: list) -> go.Figure:
    """
    Biểu đồ SHAP Beeswarm — phân tán SHAP values theo giá trị feature.
    Cho thấy mối quan hệ giữa giá trị feature cao/thấp và tác động lên/xuống.

    Returns:
        Plotly Figure
    """
    # Tính mean |SHAP| để sắp xếp
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    top_n = min(8, len(feature_names))
    top_idx = sorted_idx[:top_n]

    fig = make_subplots(rows=top_n, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02)

    for i, idx in enumerate(top_idx):
        feat_name = feature_names[idx]
        feat_values = X_data[:, idx]
        shap_vals = shap_values[:, idx]

        # Normalize feature values cho colormap
        feat_min, feat_max = feat_values.min(), feat_values.max()
        if feat_max > feat_min:
            feat_norm = (feat_values - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = np.zeros_like(feat_values)

        # Sample nếu quá nhiều điểm
        if len(shap_vals) > 200:
            sample_idx = np.random.choice(len(shap_vals), 200, replace=False)
            shap_vals_s = shap_vals[sample_idx]
            feat_norm_s = feat_norm[sample_idx]
        else:
            shap_vals_s = shap_vals
            feat_norm_s = feat_norm

        fig.add_trace(
            go.Scatter(
                x=shap_vals_s,
                y=np.random.normal(0, 0.1, len(shap_vals_s)),
                mode='markers',
                marker=dict(
                    size=5,
                    color=feat_norm_s,
                    colorscale='RdBu_r',
                    opacity=0.6,
                    showscale=(i == 0),
                    colorbar=dict(title='Feature Value<br>(normalized)')
                ),
                name=feat_name,
                showlegend=False
            ),
            row=i + 1, col=1
        )

        fig.update_yaxes(
            title_text=feat_name,
            showticklabels=False,
            row=i + 1, col=1
        )

    fig.update_layout(
        title='🐝 SHAP Beeswarm — Phân tán tác động của Features',
        template='plotly_dark',
        height=top_n * 80 + 100,
        xaxis=dict(title='SHAP value')
    )

    return fig


def get_shap_explanation_text(shap_values: np.ndarray, feature_names: list,
                              X_single: np.ndarray, prediction: float) -> str:
    """
    Tạo giải thích bằng text cho 1 prediction.

    Returns:
        String mô tả các yếu tố ảnh hưởng
    """
    abs_shap = np.abs(shap_values)
    sorted_idx = np.argsort(abs_shap)[::-1]

    lines = [f"**Giá dự đoán: {prediction:,.0f} VNĐ**\n", "**Các yếu tố chính ảnh hưởng:**\n"]

    for i, idx in enumerate(sorted_idx[:5]):
        feat = feature_names[idx]
        val = X_single[idx]
        shap_val = shap_values[idx]
        direction = "📈 đẩy giá LÊN" if shap_val > 0 else "📉 kéo giá XUỐNG"
        lines.append(f"{i+1}. **{feat}** = {val:,.2f} → {direction} {abs(shap_val):,.0f} VNĐ")

    return "\n".join(lines)
