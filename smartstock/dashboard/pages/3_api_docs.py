"""Page 3 — API Documentation"""

import streamlit as st

from smartstock.dashboard.utils import load_css, render_sidebar_status

load_css()
render_sidebar_status()

st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-title">API Documentation</div>
      <div class="ss-hero-subtitle">
        Integrate SmartStock forecasting and optimisation into your own services via REST.
        <span class="ss-badge ss-badge-warning" style="margin-left:0.75rem;vertical-align:middle;">
          Coming Soon
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="ss-alert ss-alert-info">
      The REST API is planned for Phase 2. This page documents the intended interface so
      you can start building integrations today. All endpoint shapes are final.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Base URL & Auth ───────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "<h3 style=\"font-family:'Alice',Georgia,serif;color:#c0c8d4;font-size:1.1rem;\">Base URL</h3>",
        unsafe_allow_html=True,
    )
    st.code("https://api.smartstock.app/v1", language="text")

with c2:
    st.markdown(
        "<h3 style=\"font-family:'Alice',Georgia,serif;color:#c0c8d4;font-size:1.1rem;\">Authentication</h3>",
        unsafe_allow_html=True,
    )
    st.code("X-API-Key: sk_live_xxxxxxxxxxxxxxxxxxxx", language="text")

st.markdown("<br>", unsafe_allow_html=True)

# ── Endpoints ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style=\"font-family:'Alice',Georgia,serif;color:#c0c8d4;font-size:1.2rem;margin-bottom:1rem;\">Endpoints</h2>",
    unsafe_allow_html=True,
)

ENDPOINTS = [
    {
        "method": "GET",
        "path": "/health",
        "description": "Health check — returns API status and version.",
        "curl": 'curl -H "X-API-Key: YOUR_KEY" https://api.smartstock.app/v1/health',
        "python": 'import httpx\nr = httpx.get("https://api.smartstock.app/v1/health",\n              headers={"X-API-Key": "YOUR_KEY"})\nprint(r.json())  # {"status": "ok", "version": "1.0.0"}',
        "response": '{"status": "ok", "version": "1.0.0"}',
    },
    {
        "method": "POST",
        "path": "/forecast",
        "description": "Generate a demand forecast for a single store-item time series. "
        "Returns point forecasts with upper/lower confidence bounds.",
        "curl": """curl -X POST https://api.smartstock.app/v1/forecast \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \\
  -d '{"data":[{"date":"2023-01-01","sales":42}],"model":"prophet","periods":30}'""",
        "python": """import httpx
r = httpx.post(
    "https://api.smartstock.app/v1/forecast",
    json={"data": [{"date": "2023-01-01", "sales": 42}],
          "model": "prophet", "periods": 30},
    headers={"X-API-Key": "YOUR_KEY"}
)
print(r.json())""",
        "response": """{
  "forecast": [
    {"date":"2023-02-01","forecast":41.2,"ci_lower":35.0,"ci_upper":47.4}
  ],
  "model_used": "prophet",
  "metrics": {"mae":3.1,"rmse":4.2,"mape":7.8,"r2":0.91}
}""",
    },
    {
        "method": "POST",
        "path": "/optimize",
        "description": "Calculate EOQ, Safety Stock, and Reorder Point from a forecast series. "
        "Returns per-period order quantities.",
        "curl": """curl -X POST https://api.smartstock.app/v1/optimize \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \\
  -d '{"forecast_series":[41.2,38.7],"ordering_cost":50,"holding_cost_per_period":0.5}'""",
        "python": """import httpx
r = httpx.post(
    "https://api.smartstock.app/v1/optimize",
    json={"forecast_series": [41.2, 38.7, 44.1],
          "ordering_cost": 50.0,
          "holding_cost_per_period": 0.5,
          "lead_time_periods": 7,
          "service_level": 0.95},
    headers={"X-API-Key": "YOUR_KEY"}
)
print(r.json())""",
        "response": """{
  "results": [
    {"period":0,"expected_demand":41.2,"eoq":91,
     "safety_stock":12,"reorder_point":300,"total_order_quantity":110}
  ]
}""",
    },
    {
        "method": "POST",
        "path": "/abc-analysis",
        "description": "Classify products into A/B/C tiers using the Pareto principle. "
        "A items (top 20%) typically drive 80% of inventory value.",
        "curl": """curl -X POST https://api.smartstock.app/v1/abc-analysis \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \\
  -d '{"items":[{"item_id":"SKU001","unit_cost":12.5,"annual_demand":1200}]}'""",
        "python": """import httpx
r = httpx.post(
    "https://api.smartstock.app/v1/abc-analysis",
    json={"items": [{"item_id": "SKU001", "unit_cost": 12.5, "annual_demand": 1200}]},
    headers={"X-API-Key": "YOUR_KEY"}
)
print(r.json())""",
        "response": """{
  "results": [
    {"item_id":"SKU001","annual_value":15000.0,
     "cumulative_value_pct":0.72,"abc_category":"A"}
  ]
}""",
    },
]

METHOD_COLORS = {
    "GET": ("#4ade80", "rgba(74,222,128,0.08)"),
    "POST": ("#60a5fa", "rgba(96,165,250,0.08)"),
}

for ep in ENDPOINTS:
    color, bg = METHOD_COLORS.get(ep["method"], ("#c0c8d4", "rgba(192,200,212,0.08)"))
    with st.container():
        st.markdown(
            f"""
            <div class="ss-card" style="margin-bottom:0.5rem;">
              <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
                <span style="background:{bg};border:1px solid {color}40;color:{color};
                             border-radius:6px;padding:0.2rem 0.6rem;
                             font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">
                  {ep["method"]}
                </span>
                <code style="color:#c0c8d4;font-size:0.95rem;">{ep["path"]}</code>
              </div>
              <p style="color:#636878;font-size:0.85rem;margin:0;">{ep["description"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tab_curl, tab_py, tab_resp = st.tabs(["cURL", "Python", "Response"])
        with tab_curl:
            st.code(ep["curl"], language="bash")
        with tab_py:
            st.code(ep["python"], language="python")
        with tab_resp:
            st.code(ep["response"], language="json")
        st.markdown("<br>", unsafe_allow_html=True)

# ── Error codes ───────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style=\"font-family:'Alice',Georgia,serif;color:#c0c8d4;font-size:1.2rem;margin:1.5rem 0 1rem;\">Error Codes</h2>",
    unsafe_allow_html=True,
)
for code, name, desc in [
    ("400", "Bad Request", "Missing or invalid request fields."),
    ("401", "Unauthorized", "Missing or invalid API key."),
    ("422", "Unprocessable Entity", "Request body failed schema validation."),
    (
        "429",
        "Too Many Requests",
        "Rate limit exceeded. Retry after the indicated delay.",
    ),
    ("500", "Internal Server Error", "Unexpected server-side error."),
]:
    st.markdown(
        f"""<div style="display:flex;gap:1rem;padding:0.6rem 0;border-bottom:1px solid #2a2d35;">
          <code style="color:#fbbf24;min-width:3rem;">{code}</code>
          <span style="color:#c0c8d4;min-width:10rem;font-size:0.875rem;">{name}</span>
          <span style="color:#636878;font-size:0.875rem;">{desc}</span>
        </div>""",
        unsafe_allow_html=True,
    )
