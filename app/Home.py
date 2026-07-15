"""Entry point: defines navigation and runs the selected page."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import nav  # noqa: E402

st.set_page_config(page_title="Finger conductivity simulator", page_icon="🩻", layout="wide")

st.navigation(nav.ordered()).run()
