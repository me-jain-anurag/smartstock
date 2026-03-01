"""Page 2 — References & Citations"""

import re
from pathlib import Path

import streamlit as st

from smartstock.dashboard.utils import load_css, render_sidebar_status

load_css()
render_sidebar_status()

st.markdown(
    """
    <div class="ss-hero">
      <div class="ss-hero-title">References</div>
      <div class="ss-hero-subtitle">
        Every formula, library, and algorithm in SmartStock is peer-reviewed and cited.
        Verify our math — all sources are linked directly.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="ss-alert ss-alert-info">
      <strong>New to inventory terms?</strong>
      Plain-language definitions of EOQ, Safety Stock, ROP, ABC Analysis,
      SARIMA, Prophet, MAE, MAPE, and R² are included in each section below.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Load references.md ────────────────────────────────────────────────────────
_REFS_PATH = Path(__file__).parents[3] / "references.md"

if not _REFS_PATH.exists():
    st.error(
        f"references.md not found. Expected at: `{_REFS_PATH.resolve()}`\n\n"
        "Make sure you are running the app from the project root directory."
    )
    st.stop()

raw = _REFS_PATH.read_text(encoding="utf-8")


def _linkify(text: str) -> str:
    """
    Convert bare https?:// URLs in text to clickable markdown links.

    Uses [^\\s<>"()\\[\\]]+ (greedy) so the match includes dots inside URLs
    (e.g. pypi.org, github.com). Trailing punctuation stripped separately.
    Already-linked markdown ([text](url)) is skipped by the (?<!\\() lookbehind.
    """
    _URL_RE = re.compile(r'(?<!\()https?://[^\s<>"()\[\]]+')

    def _replace(m: re.Match[str]) -> str:
        url = m.group(0).rstrip(".,;:!?)'\"")
        if "doi.org" in url:
            doi_id = url.split("doi.org/")[-1]
            return f"[doi:{doi_id}]({url})"
        return f"[{url}]({url})"

    return _URL_RE.sub(_replace, text)


# ── Parse & render sections as plain markdown (no expanders) ──────────────────
# Avoids Streamlit expander icon-rendering issues (Material Icons CDN failures)
# References is a read-once document — sections work better as scrollable content.
sections = re.split(r"\n(?=## )", raw)
intro = sections[0] if sections else ""

if intro.strip():
    # Drop the top-level H1, render the intro paragraph
    intro_body = re.sub(r"^# .+\n", "", intro).strip()
    st.markdown(_linkify(intro_body))
    st.divider()

for section in sections[1:]:
    lines = section.strip().split("\n")
    heading = lines[0].lstrip("#").strip()
    body = "\n".join(lines[1:]).strip()
    # Render heading as styled HTML to use Alice font
    st.markdown(
        f"<h2 style=\"font-family:'Alice',Georgia,serif;font-size:1.2rem;"
        f'color:#c0c8d4;margin:1.5rem 0 0.75rem;">{heading}</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(_linkify(body))
    st.divider()
