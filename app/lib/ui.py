"""Shared visual chrome: injected theme, page headers, tissue legend rail."""

from __future__ import annotations

import streamlit as st

from lib import palette

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --ink: {palette.INK};
  --muted: {palette.MUTED};
  --paper: {palette.PAPER};
  --surface: {palette.SURFACE};
  --line: {palette.LINE};
  --arterial: {palette.ARTERIAL};
}}

html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.01em; }}

/* Eyebrow / kicker above titles */
.kicker {{
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--arterial);
  margin-bottom: 0.15rem;
}}
.page-title {{ font-size: 2.1rem; font-weight: 700; margin: 0 0 0.15rem 0; color: var(--ink); }}
.page-sub {{ color: var(--muted); font-size: 0.98rem; max-width: 60ch; }}

/* Hero cross-section rule under the hub headline */
.hero-rule {{ height: 3px; width: 64px; background: var(--arterial); margin: 0.9rem 0 1.4rem; border-radius: 2px; }}

/* Navigation cards on the hub */
.nav-card {{
  display: block; text-decoration: none; height: 100%;
  border: 1px solid var(--line); border-radius: 14px; background: var(--surface);
  padding: 1.15rem 1.2rem; transition: border-color .15s ease, transform .15s ease;
}}
.nav-card:hover {{ border-color: var(--arterial); transform: translateY(-2px); }}
.nav-index {{ font-family: 'IBM Plex Mono', monospace; color: var(--arterial); font-size: 0.8rem; letter-spacing: 0.1em; }}
.nav-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.18rem; color: var(--ink); margin: 0.35rem 0 0.25rem; }}
.nav-desc {{ color: var(--muted); font-size: 0.9rem; line-height: 1.45; }}

/* Tissue spectrum rail (the page signature) */
.rail {{ display: flex; flex-direction: column; gap: 0.4rem; }}
.rail-row {{ display: flex; align-items: center; gap: 0.55rem; font-size: 0.85rem; color: var(--ink); }}
.rail-swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.12); flex: 0 0 auto; }}

.stApp {{ background: var(--paper); }}
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--line); }}
div.stButton > button[kind="primary"] {{ background: var(--arterial); border: none; }}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, subtitle: str) -> None:
    inject_theme()
    st.markdown(
        f'<div class="kicker">{kicker}</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="hero-rule"></div>'
        f'<div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )
    st.write("")


def tissue_rail() -> None:
    rows = "".join(
        f'<div class="rail-row"><span class="rail-swatch" style="background:{palette.TISSUE_COLORS[t]}"></span>{palette.TISSUE_LABELS[t]}</div>'
        for t in palette.LEGEND_ORDER
    )
    st.markdown(f'<div class="rail">{rows}</div>', unsafe_allow_html=True)
