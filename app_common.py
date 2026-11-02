"""Shared resources for the Streamlit pages: model, metrics files and server-wide logs."""
import collections
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import inference  # noqa: E402
from config import CLASSES, MODELS_DIR  # noqa: E402,F401


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return inference.load_model()


@st.cache_resource
def prediction_log():
    """Every prediction served by this server (shared by all visitors)."""
    return collections.deque(maxlen=1000)


@st.cache_resource
def resource_log():
    """CPU / memory samples taken by the monitor page."""
    return collections.deque(maxlen=300)


@st.cache_data
def load_metrics(name):
    path = MODELS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None
