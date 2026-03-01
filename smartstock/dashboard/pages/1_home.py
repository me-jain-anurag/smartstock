"""Page 1 — Home / About"""

import streamlit as st

from smartstock.dashboard.utils import load_css, render_sidebar_status

load_css()
render_sidebar_status()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-title">SmartStock</div>
      <div class="ss-hero-subtitle">
        AI-powered demand forecasting &amp; inventory optimisation for modern supply chains.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="font-family:'Space Grotesk',system-ui,sans-serif;
               font-size:1rem;color:#9aa0b0;max-width:720px;line-height:1.75;">
      SmartStock helps small and medium businesses make smarter inventory decisions
      by combining classical inventory theory (EOQ, Safety Stock, Reorder Points)
      with modern machine learning forecasting (Prophet, SARIMA).
      Every formula is cited, every decision is explainable.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature Cards ─────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.25rem;color:#c0c8d4;margin-bottom:1.25rem;\">What SmartStock does</h2>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
cards = [
    (
        col1,
        "📈",
        "Demand Forecasting",
        "Prophet and SARIMA models generate accurate daily demand forecasts with "
        "confidence intervals — handling seasonality, holidays, and trends automatically.",
    ),
    (
        col2,
        "⚙️",
        "EOQ Optimisation",
        "Dynamic Economic Order Quantity adapts to forecasted demand each period, "
        "minimising total inventory cost while respecting supplier batch constraints.",
    ),
    (
        col3,
        "🏷️",
        "ABC Analysis",
        "Pareto-based product prioritisation classifies your catalogue into A, B, and C "
        "tiers — so you focus compute and attention where it matters most.",
    ),
]
for col, icon, title, desc in cards:
    with col:
        st.markdown(
            f"""
            <div class="ss-feature-card">
              <span class="ss-feature-icon">{icon}</span>
              <div class="ss-feature-title">{title}</div>
              <div class="ss-feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.25rem;color:#c0c8d4;margin-bottom:1.25rem;\">How it works</h2>",
    unsafe_allow_html=True,
)

steps_cols = st.columns(4)
steps = [
    ("1", "Upload", "Drag &amp; drop your sales CSV. Column names are auto-detected."),
    ("2", "Forecast", "Choose a model and horizon. Results appear in seconds."),
    ("3", "Optimise", "Feed the forecast into EOQ to get per-period order quantities."),
    ("4", "Prioritise", "Run ABC analysis to identify your highest-value products."),
]
for col, (num, title, desc) in zip(steps_cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="ss-card" style="text-align:center;">
              <div style="font-family:'Alice',Georgia,serif;font-size:2rem;
                          color:#4a5162;margin-bottom:0.5rem;">{num}</div>
              <div style="font-family:'Alice',Georgia,serif;font-size:1rem;
                          color:#c0c8d4;margin-bottom:0.4rem;">{title}</div>
              <div style="font-size:0.8rem;color:#636878;line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.25rem;color:#c0c8d4;margin-bottom:1rem;\">Built with</h2>",
    unsafe_allow_html=True,
)
badges = [
    "Prophet",
    "SARIMA / Statsmodels",
    "pandas",
    "NumPy",
    "SciPy",
    "Streamlit",
    "Plotly",
    "Hypothesis (PBT)",
    "FastAPI (coming soon)",
]
st.markdown(
    '<div style="line-height:2.5;">'
    + " ".join(f'<span class="ss-tech-badge">{b}</span>' for b in badges)
    + "</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── References callout ─────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ss-card" style="display:flex;align-items:center;gap:1.25rem;">
      <span style="font-size:2rem;">📚</span>
      <div>
        <div style="font-family:'Alice',Georgia,serif;font-size:1rem;
                    color:#c0c8d4;margin-bottom:0.25rem;">Peer-reviewed foundations</div>
        <div style="font-size:0.85rem;color:#636878;">
          Every formula, library, and algorithm in SmartStock is backed by peer-reviewed research.
          Unfamiliar with EOQ, SARIMA, or ABC Analysis?
          The <strong style="color:#8a94a6;">References</strong> page has plain-language definitions
          alongside full citations so you can verify the math.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.page_link("pages/2_references.py", label="Go to References →", icon="📚")
