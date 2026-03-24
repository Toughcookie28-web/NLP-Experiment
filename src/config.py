import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR        = ROOT / "data"
ARTICLES_PATH   = DATA_DIR / "news" / "articles.jsonl"
RESPONSES_PATH  = DATA_DIR / "raw" / "responses.jsonl"
RESULTS_PATH    = DATA_DIR / "structured" / "results.csv"
FIGURES_DIR     = DATA_DIR / "figures"

# ── API ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]   # set in shell before running
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Models ─────────────────────────────────────────────────────────────────
# Fill in SMALL_MODEL_ID before running. Pick any OpenRouter model with
# adequate context window (e.g. "meta-llama/llama-3.1-8b-instruct").
GPT4O_ID    = "openai/gpt-4o"
O3MINI_ID   = "openai/o3-mini"
SMALL_MODEL_ID = "FILL_IN_BEFORE_RUNNING"   # <-- fill this in

MODELS = [GPT4O_ID, O3MINI_ID, SMALL_MODEL_ID]

# o3-mini does not accept temperature — each model has its own call params
MODEL_PARAMS = {
    GPT4O_ID:      {"temperature": 0.7},
    O3MINI_ID:     {},                        # no temperature for reasoning models
    SMALL_MODEL_ID: {"temperature": 0.7},
}

# ── Experiment settings ────────────────────────────────────────────────────
SAMPLES_PER_CONDITION = 20
EXTRACTOR_MODEL = "openai/gpt-4o-mini"
EXTRACTOR_TEMPERATURE = 0

# ── Analysis ───────────────────────────────────────────────────────────────
# Coverage below this threshold causes analysis to abort and warn
COVERAGE_THRESHOLD = 0.95
