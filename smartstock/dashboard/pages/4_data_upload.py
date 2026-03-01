"""Page 4 — Data Upload"""

import streamlit as st

from smartstock.dashboard.service import REQUIRED_COLS, load_and_validate_csv
from smartstock.dashboard.utils import load_css, render_sidebar_status

load_css()
render_sidebar_status()

st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-title">Data Upload</div>
      <div class="ss-hero-subtitle">
        Upload your sales CSV to begin forecasting and optimisation.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cols_str = " &nbsp;·&nbsp; ".join(f"<code>{c}</code>" for c in REQUIRED_COLS)
st.markdown(
    f"""
    <div class="ss-alert ss-alert-info">
      <strong>Required columns:</strong> {cols_str}<br>
      <span style="font-size:0.82rem;opacity:0.8;">
        Common aliases (<code>Date</code>, <code>Sales</code>, <code>qty</code>,
        <code>item_id</code>) are auto-detected and remapped automatically.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop your CSV here, or click to browse",
    type=["csv"],
    help=f"Required columns: {', '.join(REQUIRED_COLS)}",
    key="csv_uploader",
)

if uploaded is not None:
    with st.spinner("Validating your file…"):
        df, warnings, error_msg = load_and_validate_csv(uploaded)

    if error_msg is not None:
        st.markdown(
            f"""
            <div class="ss-card" style="border-color:#f87171;background:rgba(248,113,113,0.05);">
              <div style="font-size:0.8rem;font-weight:700;letter-spacing:0.07em;
                           text-transform:uppercase;color:#f87171;margin-bottom:0.6rem;">
                ❌ Validation Failed
              </div>
              <div style="font-size:0.875rem;color:#9aa0b0;line-height:1.7;">
                {error_msg}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["raw_df"] = None
        st.session_state["column_warnings"] = []
        st.stop()

    if warnings:
        st.session_state["column_warnings"] = warnings
        warning_lines = "<br>".join(f"• {w}" for w in warnings)
        st.markdown(
            f"""
            <div class="ss-card" style="border-color:#fbbf24;background:rgba(251,191,36,0.05);">
              <div style="font-size:0.8rem;font-weight:700;letter-spacing:0.07em;
                           text-transform:uppercase;color:#fbbf24;margin-bottom:0.6rem;">
                ⚠ Column Auto-Detection
              </div>
              <div style="font-size:0.875rem;color:#9aa0b0;line-height:1.8;">
                We remapped the following columns:<br>{warning_lines}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_ok, col_no = st.columns([1, 5])
        with col_ok:
            confirmed = st.button(
                "✓ Looks correct", type="primary", key="confirm_mapping"
            )
        with col_no:
            rejected = st.button("✗ That's wrong — let me rename", key="reject_mapping")

        if rejected:
            st.error(
                f"Please rename your columns to exactly: `{', '.join(REQUIRED_COLS)}`  \n"
                "Then re-upload the file."
            )
            st.session_state["raw_df"] = None
            st.stop()
        if not confirmed:
            st.info("Confirm the column mapping above to proceed.")
            st.stop()
    else:
        st.session_state["column_warnings"] = []

    st.session_state["raw_df"] = df
    st.session_state["forecast_result"] = None

    st.markdown(
        """
        <div class="ss-alert ss-alert-success">
          <strong>✓ Data loaded successfully.</strong>
          Available across all pages in this session.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.2rem;color:#c0c8d4;margin-bottom:1rem;\">Dataset Summary</h2>",
        unsafe_allow_html=True,
    )
    assert df is not None

    date_min = df["date"].min()
    date_max = df["date"].max()
    days_span = (date_max - date_min).days

    m1, m2, m3, m4, m5 = st.columns(5)
    for col, label, val in [
        (m1, "Rows", f"{len(df):,}"),
        (m2, "Stores", str(df["store"].nunique())),
        (m3, "Items", str(df["item"].nunique())),
        (m4, "Date Range", f"{days_span:,} days"),
        (m5, "Missing Sales", str(df["sales"].isna().sum())),
    ]:
        with col:
            st.metric(label=label, value=val)

    st.markdown(
        f'<p style="font-size:0.78rem;color:#636878;margin-top:0.5rem;">'
        f'{date_min.strftime("%d %b %Y")} → {date_max.strftime("%d %b %Y")}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.2rem;color:#c0c8d4;margin:1.5rem 0 0.5rem;\">Preview — Required columns (first 10 rows)</h2>",
        unsafe_allow_html=True,
    )
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLS]
    st.dataframe(
        df[REQUIRED_COLS].head(10).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    if extra_cols:
        st.caption(
            f"ℹ️ Your file also has {len(extra_cols)} extra column(s): "
            f"`{'`, `'.join(extra_cols)}`. "
            "SmartStock only reads `date`, `store`, `item`, `sales` — extras are safely ignored."
        )

    try:
        import plotly.graph_objects as go

        daily = df.groupby("date")["sales"].sum().reset_index()
        daily.columns = ["date", "total_sales"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["total_sales"],
                mode="lines",
                name="Daily Sales",
                line=dict(color="#c0c8d4", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(192,200,212,0.06)",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#16181d",
            plot_bgcolor="#16181d",
            font=dict(family="Space Grotesk", color="#9aa0b0"),
            xaxis=dict(gridcolor="#2a2d35", title="Date"),
            yaxis=dict(gridcolor="#2a2d35", title="Total Units Sold"),
            margin=dict(l=16, r=16, t=16, b=16),
            hovermode="x unified",
        )
        st.markdown(
            "<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.2rem;color:#c0c8d4;margin:1.5rem 0 0.25rem;\">Daily Total Sales</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.82rem;color:#636878;margin-bottom:0.75rem;">'
            "Total units sold across all stores and items each day. "
            "Hover over the chart to see exact values."
            "</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render sales chart: {e}")


elif st.session_state.get("raw_df") is not None:
    df = st.session_state["raw_df"]
    st.success(
        f"**Data already loaded** — {len(df):,} rows, "
        f"{df['store'].nunique()} stores, {df['item'].nunique()} items.  \n"
        "Upload a new file above to replace it."
    )
    extra_cols_cached = [c for c in df.columns if c not in REQUIRED_COLS]
    st.dataframe(
        df[REQUIRED_COLS].head(10).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    if extra_cols_cached:
        st.caption(
            f"ℹ️ {len(extra_cols_cached)} extra column(s) in the loaded file are ignored: "
            f"`{'`, `'.join(extra_cols_cached)}`"
        )

else:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 2rem;border:1px dashed #2a2d35;
                    border-radius:14px;margin-top:1rem;">
          <div style="font-size:2.5rem;margin-bottom:0.75rem;">📂</div>
          <div style="font-family:'Alice',Georgia,serif;font-size:1.1rem;
                      color:#c0c8d4;margin-bottom:0.4rem;">No file uploaded yet</div>
          <div style="font-size:0.85rem;color:#636878;">
            Upload a CSV above to begin.
            Use <code>data/raw/sample_quick.csv</code> if you need sample data.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
