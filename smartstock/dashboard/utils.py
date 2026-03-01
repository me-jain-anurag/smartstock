"""
SmartStock Dashboard — Shared Utilities
========================================
Every page imports from here to get consistent CSS, sidebar, and data status.
"""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_CSS_PATH = Path(__file__).parent / "style.css"


def load_css() -> None:
    """Inject the global dark-mode stylesheet and fix Material Icon rendering."""
    if _CSS_PATH.exists():
        st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
    fix_material_icons()


def fix_material_icons() -> None:
    """
    Suppress Streamlit's Material Icon ligature text when the icon font fails to load.

    Streamlit renders icons like 'keyboard_double_arrow_left' and 'arrow_right' as
    text content using the Material Icons ligature system. When the Google Fonts CDN
    is blocked or slow, the raw text appears instead of the symbol.

    CSS cannot reliably target these elements because they are rendered by Streamlit's
    React/Emotion framework with dynamic class names. This function injects JavaScript
    into window.parent (the Streamlit app's top-level document) to locate and hide
    elements whose sole text content is a known Material Icon name.

    Called from load_css() so every page gets the fix automatically.
    """
    ICON_NAMES = [
        "keyboard_double_arrow_left",
        "keyboard_double_arrow_right",
        "arrow_right",
        "arrow_drop_down",
        "arrow_drop_up",
        "expand_more",
        "expand_less",
        "chevron_right",
        "chevron_left",
        "arrow_upward",
        "arrow_downward",
        "unfold_more",
        "unfold_less",
        "close",
        "menu",
        "more_vert",
        "more_horiz",
    ]
    icon_list = "', '".join(ICON_NAMES)

    components.html(
        f"""
        <script>
        (function() {{
            var ICONS = new Set(['{icon_list}']);

            function clean(doc) {{
                if (!doc) return;

                // 1. Hide sidebar collapse/expand button by data-testid
                var collapse = doc.querySelector('[data-testid="collapsedControl"]');
                if (collapse) {{ collapse.style.display = 'none'; }}

                // 2. Scan buttons and spans whose ONLY content is an icon name
                var els = doc.querySelectorAll('button, span, i');
                els.forEach(function(el) {{
                    var txt = (el.textContent || '').trim();
                    if (ICONS.has(txt)) {{
                        el.style.fontSize  = '0';
                        el.style.width     = '0';
                        el.style.height    = '0';
                        el.style.overflow  = 'hidden';
                        el.style.display   = 'inline-block';
                        el.setAttribute('aria-hidden', 'true');
                    }}
                }});
            }}

            function run() {{
                try {{ clean(window.parent.document); }} catch(e) {{}}
                try {{ clean(document); }} catch(e) {{}}
            }}

            // Run immediately and at intervals as Streamlit re-renders
            run();
            setTimeout(run, 300);
            setTimeout(run, 1000);
            setTimeout(run, 2500);

            // Also watch for Streamlit re-renders that recreate icon elements
            try {{
                var observer = new MutationObserver(function(mutations) {{
                    // Debounce to avoid hammering on every mutation
                    clearTimeout(window._ssIconTimer);
                    window._ssIconTimer = setTimeout(run, 150);
                }});
                observer.observe(window.parent.document.body, {{
                    childList: true, subtree: true
                }});
            }} catch(e) {{ /* cross-origin guard */ }}
        }})();
        </script>
        """,
        height=0,
    )


def render_sidebar_status() -> None:
    """
    Render SmartStock branding + live data status in the sidebar.
    Call from every page so the sidebar looks consistent everywhere.
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding:1rem 0 1.25rem;">
              <span style="
                font-family:'Alice',Georgia,serif;
                font-size:1.35rem;color:#c0c8d4;letter-spacing:-0.01em;
              ">📦 SmartStock</span>
              <p style="
                font-family:'Space Grotesk',system-ui,sans-serif;
                font-size:0.72rem;color:#636878;margin-top:0.2rem;
              ">AI-powered inventory intelligence</p>
            </div>
            <hr style="border-color:#2a2d35;margin:0 0 1rem;">
            """,
            unsafe_allow_html=True,
        )

        # ── Data status ───────────────────────────────────────────────────────
        df = st.session_state.get("raw_df")
        if df is not None:
            rows = len(df)
            stores = df["store"].nunique()
            items = df["item"].nunique()
            st.markdown(
                f"""
                <div style="
                  background:rgba(74,222,128,0.06);
                  border:1px solid rgba(74,222,128,0.2);
                  border-radius:8px;padding:0.75rem 1rem;
                  margin-bottom:1rem;
                ">
                  <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.1em;
                               text-transform:uppercase;color:#4ade80;margin-bottom:0.3rem;">
                    ✓ Data Loaded
                  </div>
                  <div style="font-size:0.78rem;color:#9aa0b0;">
                    {rows:,} rows · {stores} stores · {items} items
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="
                  background:rgba(251,191,36,0.06);
                  border:1px solid rgba(251,191,36,0.2);
                  border-radius:8px;padding:0.75rem 1rem;
                  margin-bottom:1rem;
                ">
                  <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.1em;
                               text-transform:uppercase;color:#fbbf24;margin-bottom:0.25rem;">
                    ⚠ No Data
                  </div>
                  <div style="font-size:0.78rem;color:#9aa0b0;">
                    Upload a CSV on the Data Upload page.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Version ───────────────────────────────────────────────────────────
        st.markdown(
            """
            <p style="font-size:0.68rem;color:#4a5162;
                      position:fixed;bottom:1.5rem;left:1.25rem;">
              SmartStock v0.1.0-alpha
            </p>
            """,
            unsafe_allow_html=True,
        )
