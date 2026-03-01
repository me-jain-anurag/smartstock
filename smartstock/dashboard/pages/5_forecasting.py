"""Page 5 — Demand Forecasting"""

import plotly.graph_objects as go
import streamlit as st

from smartstock.dashboard.service import (
    dataframe_to_csv_bytes,
    get_items,
    get_stores,
    run_forecast,
)
from smartstock.dashboard.utils import load_css, render_sidebar_status

load_css()
render_sidebar_status()

st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-title">Demand Forecasting</div>
      <div class="ss-hero-subtitle">
        Generate item-level demand forecasts using Prophet, SARIMA, or the Naive baseline.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Terms callout ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ss-alert ss-alert-info" style="font-size:0.82rem;">
      <strong>New to forecasting terms?</strong>
      MAE, MAPE, R², SARIMA, and Prophet are all explained in plain language on the
      <a href="/references" style="color:#60a5fa;">References page.</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state guard ───────────────────────────────────────────────────────
if st.session_state.get("raw_df") is None:
    st.markdown(
        """
        <div class="ss-card" style="text-align:center;padding:3rem;">
          <div style="font-size:2rem;margin-bottom:0.75rem;">📂</div>
          <div style="font-family:'Alice',Georgia,serif;font-size:1.1rem;
                      color:#c0c8d4;margin-bottom:0.5rem;">No data loaded</div>
          <div style="font-size:0.875rem;color:#636878;">
            Upload a CSV on the Data Upload page first.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_data_upload.py", label="Go to Data Upload →", icon="📂")
    st.stop()

df = st.session_state["raw_df"]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:0.7rem;font-weight:600;letter-spacing:0.1em;"
        "text-transform:uppercase;color:#8a94a6;margin:0.5rem 0;'>Forecast Settings</p>",
        unsafe_allow_html=True,
    )
    stores = get_stores(df)
    store_id = st.selectbox("Store", stores, key="fc_store")
    items = get_items(df, store_id=store_id)
    item_id = st.selectbox("Item", items, key="fc_item")
    model_name = st.selectbox(
        "Model",
        ["prophet", "sarima", "naive"],
        format_func=lambda x: {
            "prophet": "Prophet (ML — best for seasonality)",
            "sarima": "SARIMA (statistical — good for trends)",
            "naive": "Naive (baseline — last period repeated)",
        }[x],
        key="fc_model",
    )
    periods = st.slider(
        "Forecast horizon (days)", min_value=7, max_value=180, value=30, step=7
    )
    run_btn = st.button("Run Forecast", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(
        f"Training {model_name.title()} for Store {store_id} · Item {item_id}…"
    ):
        try:
            result = run_forecast(df, store_id, item_id, model_name, periods)
            st.session_state["forecast_result"] = result
        except Exception as e:
            st.error(f"**Forecast failed:** {e}")
            st.stop()

result = st.session_state.get("forecast_result")

if result is None:
    st.info(
        "Select store, item, model, and horizon in the sidebar — then click **Run Forecast**."
    )
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────
if result.metrics:
    m = result.metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, label, val, tip in [
        (
            mc1,
            "MAE",
            f"{m.get('mae', 0):.2f}",
            "Mean Absolute Error — average units off per day. Lower is better.",
        ),
        (
            mc2,
            "RMSE",
            f"{m.get('rmse', 0):.2f}",
            "Root Mean Squared Error — penalises large errors more. Lower is better.",
        ),
        (
            mc3,
            "MAPE",
            f"{m.get('mape', 0):.1f}%",
            "Mean Abs % Error — how wrong on average as a %. Under 10% is great.",
        ),
        (
            mc4,
            "R²",
            f"{m.get('r2', 0):.3f}",
            "R-squared — 1.0 = perfect. 0 = no better than guessing the mean.",
        ),
    ]:
        with col:
            st.metric(label=label, value=val, help=tip)

st.markdown("<br>", unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
fig = go.Figure()
hist = result.history
fig.add_trace(
    go.Scatter(
        x=hist.index,
        y=hist["sales"],
        name="Historical Sales",
        line=dict(color="#c0c8d4", width=1.5),
        mode="lines",
    )
)
fc = result.forecast
future_fc = fc[fc.index > hist.index[-1]]
fig.add_trace(
    go.Scatter(
        x=list(future_fc.index) + list(future_fc.index[::-1]),
        y=list(future_fc["ci_upper"]) + list(future_fc["ci_lower"][::-1]),
        fill="toself",
        fillcolor="rgba(192,200,212,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Band",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=future_fc.index,
        y=future_fc["forecast"],
        name=f"{result.model_name.title()} Forecast",
        line=dict(color="#fbbf24", width=2, dash="dot"),
        mode="lines",
    )
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#16181d",
    plot_bgcolor="#16181d",
    font=dict(family="Space Grotesk", size=12, color="#9aa0b0"),
    xaxis=dict(gridcolor="#2a2d35", title="Date"),
    yaxis=dict(gridcolor="#2a2d35", title="Units Sold"),
    legend=dict(bgcolor="rgba(22,24,29,0.8)", bordercolor="#2a2d35", borderwidth=1),
    hovermode="x unified",
    margin=dict(l=16, r=16, t=40, b=16),
    title=dict(
        text=f"Store {result.store_id} · Item {result.item_id} · {result.model_name.title()}",
        font=dict(family="Alice, Georgia, serif", size=16, color="#c0c8d4"),
    ),
)
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    label="Download Forecast CSV",
    data=dataframe_to_csv_bytes(future_fc),
    file_name=f"forecast_store{result.store_id}_item{result.item_id}.csv",
    mime="text/csv",
)
