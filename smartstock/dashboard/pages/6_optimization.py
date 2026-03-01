"""Page 6 — Inventory Optimisation (EOQ + ABC Analysis)"""

import math
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from smartstock.dashboard.service import dataframe_to_csv_bytes, run_abc, run_eoq
from smartstock.dashboard.utils import load_css, render_sidebar_status

load_css()
render_sidebar_status()

st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-title">Inventory Optimisation</div>
      <div class="ss-hero-subtitle">
        EOQ order planning and ABC product prioritisation — powered by your forecast data.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Terms callout ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ss-alert ss-alert-info" style="font-size:0.82rem;">
      <strong>New to EOQ, Safety Stock, or ABC Analysis?</strong>
      Plain-language definitions are on the
      <a href="/references" style="color:#60a5fa;">References page.</a>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_eoq, tab_abc, tab_explorer = st.tabs(
    ["EOQ Optimisation", "ABC Analysis", "EOQ Explorer"]
)

# ══════════════════════════════════════════════════════
# TAB 1 — EOQ
# ══════════════════════════════════════════════════════
with tab_eoq:
    result = st.session_state.get("forecast_result")

    if result is None:
        st.markdown(
            """
            <div class="ss-card" style="text-align:center;padding:2.5rem;">
              <div style="font-size:1.75rem;margin-bottom:0.5rem;">📈</div>
              <div style="font-family:'Alice',Georgia,serif;color:#c0c8d4;font-size:1rem;margin-bottom:0.4rem;">
                No forecast found
              </div>
              <div style="font-size:0.875rem;color:#636878;">
                Run a forecast on the Forecasting page first, then return here.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("pages/5_forecasting.py", label="Go to Forecasting →", icon="📈")
    else:
        st.markdown(
            f"""
            <div class="ss-alert ss-alert-info">
              Using forecast: <strong>Store {result.store_id} · Item {result.item_id}
              · {result.model_name.title()}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style=\"font-family:'Alice',Georgia,serif;font-size:1.1rem;color:#c0c8d4;margin-bottom:1rem;\">Parameters</h3>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            ordering_cost = st.number_input(
                "Ordering cost per order (₹)",
                min_value=0.0,
                value=50.0,
                step=5.0,
                help="Fixed cost each time an order is placed — delivery, admin, etc.",
            )
        with c2:
            holding_cost = st.number_input(
                "Holding cost per unit / period (₹)",
                min_value=0.01,
                value=0.50,
                step=0.05,
                help="Cost to store one unit for one forecast period — warehousing, insurance.",
            )
        with c3:
            lead_time = st.number_input(
                "Lead time (periods)",
                min_value=0,
                value=7,
                step=1,
                help="Periods between placing and receiving an order. Longer = more safety stock.",
            )

        c4, c5 = st.columns(2)
        with c4:
            service_level = st.slider(
                "Service Level — higher means fewer stockouts",
                0.50,
                0.999,
                0.95,
                0.005,
                format="%.3f",
                help=(
                    "Probability of never running out before the next order arrives.\n\n"
                    "0.95 = 95% stockout protection (recommended).\n"
                    "0.99 = very safe but ties up more cash in stock.\n"
                    "0.50 = you'll stockout roughly half the time."
                ),
            )
        with c5:
            batch_size = st.number_input(
                "Supplier batch size",
                min_value=1,
                value=1,
                step=1,
                help="Minimum order multiple — e.g. if supplier sells in boxes of 50, enter 50.",
            )

        if st.button("Calculate EOQ", type="primary", key="run_eoq"):
            fc = result.forecast
            future_fc = fc[fc.index > result.history.index[-1]]
            forecast_series = future_fc["forecast"]
            uncertainty = (
                (future_fc["ci_upper"] - forecast_series).clip(lower=0)
                if "ci_upper" in future_fc.columns
                else None
            )
            with st.spinner("Calculating…"):
                try:
                    eoq_df = run_eoq(
                        forecast_series=forecast_series,
                        ordering_cost=ordering_cost,
                        holding_cost_per_period=holding_cost,
                        lead_time_periods=int(lead_time),
                        uncertainty_series=uncertainty,
                        batch_size=int(batch_size),
                        service_level=service_level,
                    )
                    st.session_state["eoq_result"] = eoq_df
                except Exception as e:
                    st.error(f"**EOQ failed:** {e}")

        eoq_df = st.session_state.get("eoq_result")
        if eoq_df is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            sm1, sm2, sm3, sm4 = st.columns(4)
            for col, label, val, tip in [
                (
                    sm1,
                    "Avg EOQ",
                    f"{eoq_df['eoq'].mean():.0f} units",
                    "Average optimal order size per period.",
                ),
                (
                    sm2,
                    "Avg Safety Stock",
                    f"{eoq_df['safety_stock'].mean():.0f} units",
                    "Average buffer stock held against demand spikes.",
                ),
                (
                    sm3,
                    "Avg Reorder Point",
                    f"{eoq_df['reorder_point'].mean():.0f} units",
                    "Average stock level at which to place a new order.",
                ),
                (
                    sm4,
                    "Avg Order Qty",
                    f"{eoq_df['total_order_quantity'].mean():.0f} units",
                    "EOQ + safety stock batch-rounded to supplier minimum.",
                ),
            ]:
                with col:
                    st.metric(label=label, value=val, help=tip)

            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_bar(
                x=eoq_df.index,
                y=eoq_df["total_order_quantity"],
                name="Order Quantity",
                marker_color="#c0c8d4",
            )
            fig.add_bar(
                x=eoq_df.index,
                y=eoq_df["safety_stock"],
                name="Safety Stock",
                marker_color="#60a5fa",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#16181d",
                plot_bgcolor="#16181d",
                font=dict(family="Space Grotesk", color="#9aa0b0"),
                xaxis=dict(gridcolor="#2a2d35", title="Period"),
                yaxis=dict(gridcolor="#2a2d35", title="Units"),
                barmode="group",
                margin=dict(l=16, r=16, t=16, b=16),
                legend=dict(bgcolor="rgba(22,24,29,0.8)", bordercolor="#2a2d35"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                eoq_df.reset_index(), use_container_width=True, hide_index=True
            )
            st.download_button(
                "Download EOQ CSV",
                data=dataframe_to_csv_bytes(eoq_df),
                file_name="eoq_results.csv",
                mime="text/csv",
            )

# ══════════════════════════════════════════════════════
# TAB 2 — ABC Analysis
# ══════════════════════════════════════════════════════
with tab_abc:
    st.markdown(
        """
        <p style="font-size:0.875rem;color:#9aa0b0;margin-bottom:0.5rem;">
          ABC Analysis classifies products into three tiers by annual inventory value:
        </p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;margin-bottom:1.25rem;">
          <div class="ss-card" style="text-align:center;border-color:#fbbf2440;">
            <div class="abc-a" style="font-size:1.5rem;font-family:'Alice',Georgia,serif;">A</div>
            <div style="font-size:0.75rem;color:#636878;">Top 20% items<br>≈ 80% of value<br><em>Highest attention</em></div>
          </div>
          <div class="ss-card" style="text-align:center;border-color:#c0c8d440;">
            <div class="abc-b" style="font-size:1.5rem;font-family:'Alice',Georgia,serif;">B</div>
            <div style="font-size:0.75rem;color:#636878;">Middle 30% items<br>≈ 15% of value<br><em>Moderate attention</em></div>
          </div>
          <div class="ss-card" style="text-align:center;border-color:#63687840;">
            <div class="abc-c" style="font-size:1.5rem;font-family:'Alice',Georgia,serif;">C</div>
            <div style="font-size:0.75rem;color:#636878;">Bottom 50% items<br>≈ 5% of value<br><em>Low attention</em></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ss-alert ss-alert-warning" style="font-size:0.82rem;">
          This tab needs a <strong>separate CSV</strong> with columns:
          <code>item_id</code>, <code>unit_cost</code>, <code>annual_demand</code>.
          Use <code>data/raw/sample_abc.csv</code> to try it out.
        </div>
        """,
        unsafe_allow_html=True,
    )

    abc_source = st.radio(
        "Data source",
        ["Upload CSV", "Enter manually"],
        horizontal=True,
        key="abc_source",
        label_visibility="collapsed",
    )

    abc_df = None

    if abc_source == "Upload CSV":
        abc_file = st.file_uploader(
            "ABC analysis CSV (item_id, unit_cost, annual_demand)",
            type=["csv"],
            key="abc_uploader",
        )
        if abc_file:
            try:
                abc_df = pd.read_csv(abc_file)
                required = {"item_id", "unit_cost", "annual_demand"}
                if not required.issubset(abc_df.columns):
                    missing = required - set(abc_df.columns)
                    st.error(
                        f"Missing columns: {', '.join(f'`{c}`' for c in missing)}.  \n"
                        "Required: `item_id`, `unit_cost`, `annual_demand`.  \n"
                        "Try uploading `data/raw/sample_abc.csv`."
                    )
                    abc_df = None
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
    else:
        sample = "item_id,unit_cost,annual_demand\nSKU001,12.50,1200\nSKU002,3.00,8000\nSKU003,45.00,300"
        pasted = st.text_area(
            "Paste CSV data", value=sample, height=150, key="abc_paste"
        )
        try:
            abc_df = pd.read_csv(StringIO(pasted))
        except Exception as e:
            st.error(f"Could not parse pasted data: {e}")

    if abc_df is not None:
        if st.button("Run ABC Analysis", type="primary", key="run_abc"):
            with st.spinner("Running…"):
                try:
                    abc_result = run_abc(abc_df)
                    st.session_state["abc_result"] = abc_result
                except Exception as e:
                    st.error(f"**ABC analysis failed:** {e}")

    abc_result = st.session_state.get("abc_result")
    if abc_result is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        counts = abc_result["abc_category"].value_counts()
        ac1, ac2, ac3 = st.columns(3)
        for col, cat, label in [
            (ac1, "A", "A — Critical (top value)"),
            (ac2, "B", "B — Important"),
            (ac3, "C", "C — Low priority"),
        ]:
            with col:
                st.metric(label=label, value=f"{counts.get(cat, 0)} items")

        st.markdown("<br>", unsafe_allow_html=True)
        pie_data = (
            abc_result.groupby("abc_category")["annual_value"].sum().reset_index()
        )
        fig_pie = px.pie(
            pie_data,
            values="annual_value",
            names="abc_category",
            color="abc_category",
            color_discrete_map={"A": "#fbbf24", "B": "#c0c8d4", "C": "#4a5162"},
            template="plotly_dark",
        )
        fig_pie.update_layout(
            paper_bgcolor="#16181d",
            font=dict(family="Space Grotesk", color="#9aa0b0"),
            margin=dict(l=16, r=16, t=16, b=16),
            legend=dict(bgcolor="rgba(22,24,29,0.8)", bordercolor="#2a2d35"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        def _style_cat(val: str) -> str:
            return {
                "A": "color:#fbbf24;font-weight:600",
                "B": "color:#c0c8d4;font-weight:600",
                "C": "color:#636878",
            }.get(val, "")

        styled = abc_result.style.applymap(_style_cat, subset=["abc_category"]).format(
            {"annual_value": "₹{:,.2f}", "cumulative_value_pct": "{:.1%}"}
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ABC CSV",
            data=dataframe_to_csv_bytes(abc_result),
            file_name="abc_analysis.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════
# TAB 3 — EOQ Explorer (interactive, no CSV needed)
# ══════════════════════════════════════════════════════
with tab_explorer:
    st.markdown(
        """
        <div class="ss-alert ss-alert-info" style="font-size:0.82rem;">
          <strong>New to EOQ, Safety Stock, or Reorder Point?</strong>
          Plain-language definitions are on the
          <a href="/references" style="color:#60a5fa;">References page.</a>
          Each formula below also has a short explanation.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.875rem;color:#9aa0b0;margin:0.5rem 0 1rem;">'
        "Adjust sliders and results update live — no CSV required.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h3 style=\"font-family:'Alice',Georgia,serif;font-size:1.1rem;"
        'color:#c0c8d4;margin-bottom:1rem;">Parameters</h3>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        ex_demand = st.slider(
            "Annual demand (units)",
            100,
            100_000,
            10_000,
            100,
            help="How many units do you sell in a year? (D)",
            key="ex_demand",
        )
        ex_ordering_cost = st.slider(
            "Ordering cost per order (₹)",
            1.0,
            500.0,
            50.0,
            1.0,
            help="Fixed cost each time you place a new order — delivery, admin, paperwork. (S)",
            key="ex_ordering_cost",
        )
        ex_holding_cost = st.slider(
            "Holding cost per unit / year (₹)",
            0.1,
            50.0,
            2.0,
            0.1,
            help="Annual cost to store one unit — warehousing, insurance, spoilage. (H)",
            key="ex_holding_cost",
        )
    with col2:
        ex_lead_time = st.slider(
            "Lead time (days)",
            0,
            90,
            7,
            1,
            help="Days between placing an order and receiving it. Longer = more safety stock needed. (L)",
            key="ex_lead_time",
        )
        ex_demand_std = st.slider(
            "Daily demand variability (units std dev)",
            0.0,
            200.0,
            10.0,
            1.0,
            help="How much does daily demand fluctuate? Higher = more safety stock needed. (σ_d)",
            key="ex_demand_std",
        )
        ex_service_level = st.slider(
            "Service Level — higher means fewer stockouts",
            0.50,
            0.999,
            0.95,
            0.005,
            format="%.3f",
            help=(
                "The % probability of never running out of stock "
                "before your next order arrives.\n\n"
                "0.95 = 95% chance of no stockout (recommended for most products).\n"
                "0.99 = 99% — high safety, but much more stock held.\n"
                "0.50 = 50% — you'll stockout about half the time."
            ),
            key="ex_service_level",
        )

    # Live calculations
    ex_eoq = math.ceil(math.sqrt((2 * ex_demand * ex_ordering_cost) / ex_holding_cost))
    ex_z = stats.norm.ppf(ex_service_level)
    ex_safety_stock = (
        math.ceil(ex_z * ex_demand_std * math.sqrt(ex_lead_time))
        if ex_lead_time > 0
        else 0
    )
    ex_daily_demand = ex_demand / 365.0
    ex_rop = math.ceil(ex_daily_demand * ex_lead_time) + ex_safety_stock
    ex_annual_ordering = (ex_demand / ex_eoq) * ex_ordering_cost
    ex_annual_holding = (ex_eoq / 2) * ex_holding_cost
    ex_total_cost = ex_annual_ordering + ex_annual_holding

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style=\"font-family:'Alice',Georgia,serif;font-size:1.1rem;"
        'color:#c0c8d4;margin-bottom:1rem;">Results</h3>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    for col, label, val, tip in [
        (
            r1,
            "EOQ",
            f"{ex_eoq:,} units",
            "Optimal units to order each time — minimises total cost.",
        ),
        (
            r2,
            "Safety Stock",
            f"{ex_safety_stock:,} units",
            "Buffer held to absorb demand spikes — protects against stockouts.",
        ),
        (
            r3,
            "Reorder Point",
            f"{ex_rop:,} units",
            "When on-hand stock drops to this level, place a new order.",
        ),
        (
            r4,
            "Annual Cost",
            f"₹{ex_total_cost:,.2f}",
            "Total ordering + holding cost at the EOQ — the minimum achievable.",
        ),
        (
            r5,
            "Z-score",
            f"{ex_z:.3f}",
            f"Statistical multiplier for {ex_service_level:.1%} service level.",
        ),
    ]:
        with col:
            st.metric(label=label, value=val, help=tip)

    # Formula breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style=\"font-family:'Alice',Georgia,serif;font-size:1.1rem;"
        "color:#c0c8d4;margin-bottom:0.75rem;\">How it's calculated</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        > SmartStock applies the EOQ formula **per forecast period** (dynamic EOQ),
        > not once with a fixed annual demand. On the EOQ Optimisation tab, D changes
        > each period based on the forecast — resulting in period-specific order sizes.
        > Here the calculator uses a single annual demand for educational exploration.
        """
    )

    def _formula_card(title: str, body: str) -> None:
        st.markdown(
            f'<div class="ss-card" style="margin-bottom:1rem;">'
            f"<p style=\"font-family:'Alice',Georgia,serif;font-size:0.95rem;"
            f'color:#c0c8d4;margin:0 0 0.75rem;">{title}</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown(body)
        st.markdown(
            "<hr style='border-color:#2a2d35;margin:0.5rem 0 1.5rem;'>",
            unsafe_allow_html=True,
        )

    _formula_card(
        "EOQ — Economic Order Quantity",
        f"""**Formula:** EOQ = sqrt(2 × D × S / H)

| Symbol | Meaning | Your value |
|---|---|---|
| D | Annual demand | {ex_demand:,} units/yr |
| S | Ordering cost | ₹{ex_ordering_cost:.2f} |
| H | Holding cost | ₹{ex_holding_cost:.2f}/unit/yr |
| **EOQ** | Optimal order size | **{ex_eoq:,} units** |

*The EOQ is the quantity where ordering cost equals holding cost*
*— the sweet spot of minimum total cost.*

> Harris, F.W. (1913). How many parts to make at once.
> *Factory, The Magazine of Management*, 10(2), 135–136.
""",
    )
    _formula_card(
        "Safety Stock — Buffer against uncertainty",
        f"""**Formula:** SS = Z × daily_std × sqrt(lead_time)

| Symbol | Meaning | Your value |
|---|---|---|
| Z | Z-score for {ex_service_level:.1%} service level | {ex_z:.3f} |
| daily_std | Daily demand std dev | {ex_demand_std:.1f} units |
| lead_time | Days to receive order | {ex_lead_time} days |
| **SS** | Safety stock | **{ex_safety_stock:,} units** |

*Safety stock is the extra buffer you hold so that random demand spikes*
*don't cause stockouts.*
""",
    )
    _formula_card(
        "Reorder Point — When to place a new order",
        f"""**Formula:** ROP = (D / 365 × lead_time) + Safety Stock

| Component | Value |
|---|---|
| Daily demand | {ex_daily_demand:.2f} units/day |
| Lead time demand | {math.ceil(ex_daily_demand * ex_lead_time):,} units |
| Safety stock | {ex_safety_stock:,} units |
| **Reorder Point** | **{ex_rop:,} units** |

*When your stock drops to this level, place a new order*
*— it will arrive just before you run out.*
""",
    )

    # Cost curve
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style=\"font-family:'Alice',Georgia,serif;font-size:1.1rem;"
        'color:#c0c8d4;margin-bottom:0.5rem;">Cost Curve</h3>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.85rem;color:#636878;margin-bottom:1rem;">'
        "Total cost at different order quantities. EOQ marks the minimum.</p>",
        unsafe_allow_html=True,
    )

    q_range = np.linspace(max(1, ex_eoq // 4), ex_eoq * 4, 300)
    order_c = (ex_demand / q_range) * ex_ordering_cost
    hold_c = (q_range / 2) * ex_holding_cost
    total_c = order_c + hold_c

    fig_ex = go.Figure()
    fig_ex.add_trace(
        go.Scatter(
            x=q_range, y=total_c, name="Total Cost", line=dict(color="#c0c8d4", width=2)
        )
    )
    fig_ex.add_trace(
        go.Scatter(
            x=q_range,
            y=order_c,
            name="Ordering Cost",
            line=dict(color="#60a5fa", width=1.5, dash="dash"),
        )
    )
    fig_ex.add_trace(
        go.Scatter(
            x=q_range,
            y=hold_c,
            name="Holding Cost",
            line=dict(color="#fbbf24", width=1.5, dash="dash"),
        )
    )
    fig_ex.add_vline(
        x=ex_eoq,
        line=dict(color="#4ade80", width=1.5, dash="dot"),
        annotation_text=f"EOQ = {ex_eoq:,}",
        annotation_font=dict(color="#4ade80", family="Space Grotesk"),
    )
    fig_ex.update_layout(
        template="plotly_dark",
        paper_bgcolor="#16181d",
        plot_bgcolor="#16181d",
        font=dict(family="Space Grotesk", color="#9aa0b0"),
        xaxis=dict(gridcolor="#2a2d35", title="Order Quantity (units)"),
        yaxis=dict(gridcolor="#2a2d35", title="Annual Cost (₹)"),
        legend=dict(bgcolor="rgba(22,24,29,0.8)", bordercolor="#2a2d35", borderwidth=1),
        margin=dict(l=16, r=16, t=16, b=16),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ex, use_container_width=True)
