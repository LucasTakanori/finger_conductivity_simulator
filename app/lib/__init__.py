"""Shared helpers for the multipage finger conductivity webapp."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the scientific package importable no matter which page Streamlit runs.
_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
