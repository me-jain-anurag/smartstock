"""
SmartStock Dashboard — Main Entry Point
========================================
Run with: streamlit run smartstock/dashboard/app.py
"""

from pathlib import Path
from typing import Any

import streamlit as st

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="SmartStock",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SmartStock — AI-powered inventory management. "
        "Demand forecasting · EOQ optimisation · ABC analysis.",
    },
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "style.css"
with open(_CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "raw_df": None,  # validated DataFrame from Data Upload page
    "column_warnings": [],  # auto-detection notices
    "forecast_result": None,  # latest ForecastResult from Forecasting page
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 1rem 0 1.5rem;">
          <span style="
            font-family: 'Alice', Georgia, serif;
            font-size: 1.4rem;
            color: #c0c8d4;
            letter-spacing: -0.01em;
          ">📦 SmartStock</span>
          <p style="
            font-family: 'Space Grotesk', system-ui, sans-serif;
            font-size: 0.75rem;
            color: #636878;
            margin-top: 0.25rem;
            font-weight: 400;
          ">AI-powered inventory intelligence</p>
        </div>
        <hr style="border-color: #2a2d35; margin: 0 0 1rem;">
        """,
        unsafe_allow_html=True,
    )

    # Data status indicator
    if st.session_state["raw_df"] is not None:
        df = st.session_state["raw_df"]
        row_count = len(df)
        store_count = df["store"].nunique()
        item_count = df["item"].nunique()
        st.markdown(
            f"""
            <div style="
              background: rgba(74,222,128,0.06);
              border: 1px solid rgba(74,222,128,0.2);
              border-radius: 8px;
              padding: 0.75rem 1rem;
              margin-bottom: 1rem;
            ">
              <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;
                          text-transform:uppercase;color:#4ade80;margin-bottom:0.4rem;">
                ✓ Data Loaded
              </div>
              <div style="font-size:0.78rem;color:#9aa0b0;">
                {row_count:,} rows · {store_count} stores · {item_count} items
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
              background: rgba(251,191,36,0.06);
              border: 1px solid rgba(251,191,36,0.2);
              border-radius: 8px;
              padding: 0.75rem 1rem;
              margin-bottom: 1rem;
            ">
              <div style="font-size:0.65rem;font-weight:600;letter-spacing:0.1em;
                          text-transform:uppercase;color:#fbbf24;margin-bottom:0.3rem;">
                ⚠ No Data
              </div>
              <div style="font-size:0.78rem;color:#9aa0b0;">
                Upload a CSV on the Data Upload page to begin.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <p style="font-size:0.7rem;color:#636878;position:fixed;bottom:1.5rem;">
          SmartStock · v0.1.0-alpha
        </p>
        """,
        unsafe_allow_html=True,
    )

# ── Home redirect ─────────────────────────────────────────────────────────────
# When the user lands on app.py directly, show a minimal welcome that
# directs them to the Home page in the sidebar.
st.markdown(
    """
    <div style="
      display:flex;flex-direction:column;align-items:center;
      justify-content:center;padding:5rem 2rem;text-align:center;
    ">
      <span style="
        font-family:'Alice',Georgia,serif;
        font-size:3.5rem;color:#c0c8d4;
        letter-spacing:-0.02em;
        margin-bottom:0.5rem;
      ">📦 SmartStock</span>
      <p style="
        font-family:'Space Grotesk',system-ui,sans-serif;
        font-size:1.1rem;color:#636878;font-weight:300;
        max-width:520px;line-height:1.7;
      ">
        AI-powered demand forecasting and inventory optimisation
        for modern supply chains.
      </p>
      <p style="font-size:0.85rem;color:#4a5162;margin-top:2rem;">
        ← Use the sidebar to navigate
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
