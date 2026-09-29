"""Shared settings: folder paths, the target definition and model settings.

Every threshold that changes results lives here, so experiments are easy to reproduce.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"

# ATLAS_DEMO=1 runs the whole pipeline on synthetic data (src/make_demo_data.py) in
# separate folders, so demo output can never mix with the real Hugging Face data.
DEMO = os.getenv("ATLAS_DEMO") == "1"
_sub = "demo" if DEMO else ""

RAW_DIR = ROOT / "data" / (_sub or ".") / "raw"
CARDS_DIR = RAW_DIR / "cards"
PROCESSED_DIR = ROOT / "data" / (_sub or ".") / "processed"
MODELS_DIR = ROOT / "models" / _sub
REPORTS_DIR = ROOT / "reports" / _sub
FIGURES_DIR = REPORTS_DIR / "figures"

for folder in (RAW_DIR, CARDS_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

def _hf_token():
    # st.secrets is how Streamlit Community Cloud passes tokens (Advanced settings -> Secrets).
    # Reading it outside a deployed app (no .streamlit/secrets.toml) raises, so fall back to
    # the environment variable / .env - the app must work with no token at all either way.
    try:
        import streamlit as st
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN") or None


HF_TOKEN = _hf_token()

# ---------------------------------------------------------------------------
# Target: is a model "adopted"?
# A model is adopted when it is still being downloaded: at least
# ADOPTION_MIN_DOWNLOADS_30D downloads in the 30 days before the snapshot.
# Only models old enough to have had a fair chance are used for training.
# ---------------------------------------------------------------------------
ADOPTION_MIN_DOWNLOADS_30D = int(os.getenv("ADOPTION_MIN_DOWNLOADS_30D", "50"))
MIN_AGE_DAYS = 90      # younger models have not had time to be discovered
MAX_AGE_DAYS = 1095    # older models: popularity reflects a different era of the Hub

# Quantized/GGUF re-uploads and dedicated repackaging accounts (e.g. mradermacher) get
# downloaded by automated tools (llama.cpp, LM Studio, Ollama...) far more often than
# original models at every download threshold, which inflates the adoption rate and
# swamps genuine "did a builder's own model get used" signal. Excluded by default.
EXCLUDE_REPACKAGED = os.getenv("EXCLUDE_REPACKAGED", "1") == "1"

# Chronological split (by created date): oldest -> train, middle -> validation, newest -> test
TRAIN_FRAC, VALID_FRAC = 0.70, 0.15

RANDOM_STATE = 42

# Card text -> TF-IDF -> SVD topics
TEXT_MAX_FEATURES = 5000
TEXT_SVD_COMPONENTS = 20

# Columns that must NEVER be used as features (they are the outcome, or leak it)
LEAKY_COLUMNS = {
    "downloads_30d", "downloads_all_time", "likes", "trending_score",
    "children_model_count", "n_spaces", "is_adopted", "is_zero_download",
}
