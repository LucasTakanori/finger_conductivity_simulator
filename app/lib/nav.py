"""Central navigation definition (st.navigation / st.Page API).

Built once and cached so the entry script and the hub page share the *same*
Page objects — st.page_link only accepts pages registered with st.navigation.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import streamlit as st

# st.Page resolves relative paths against the *caller's* directory, so build
# absolute paths to the views directory to stay independent of caller location.
_VIEWS = Path(__file__).resolve().parents[1] / "views"


@lru_cache(maxsize=1)
def pages() -> dict:
    return {
        "hub": st.Page(str(_VIEWS / "hub.py"), title="Home", icon="🩻", default=True),
        "finger": st.Page(str(_VIEWS / "finger_model.py"), title="Finger model", icon="🖐️"),
        "waveform": st.Page(str(_VIEWS / "waveform.py"), title="Waveform", icon="🫀"),
        "mesh": st.Page(str(_VIEWS / "mesh_export.py"), title="Image export", icon="🧮"),
        "viewer": st.Page(str(_VIEWS / "dataset_viewer.py"), title="Dataset viewer", icon="🎞️"),
    }


def ordered() -> list:
    p = pages()
    return [p["hub"], p["finger"], p["waveform"], p["mesh"], p["viewer"]]
