"""Central configuration: paths, model names, thresholds and LLM settings.

Thresholds start at the values in the implementation workflow. Values tuned on labelled
data (reranker calibration in ai.build_index, the classifier gate in ai.classify --train,
the duplicate threshold check in ai.eval.eval_retrieval --tune) are written to
models/tuned_thresholds.json and override the defaults below at import time.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if load_dotenv:
    load_dotenv(ROOT / ".env")

# ----------------------------------------------------------------------------- dataset (it-ticket-dataset)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELS_DIR = DATA_DIR / "labels"
PROCESSED_DIR = DATA_DIR / "processed"

HISTORICAL_TICKETS = RAW_DIR / "historical_tickets.csv"   # index + classifier training
TEST_TICKETS = RAW_DIR / "test_tickets.csv"               # new tickets, input fields only
KB_ARTICLES = RAW_DIR / "kb_articles.csv"                 # approved + draft KB articles
TEST_LABELS = LABELS_DIR / "test_labels.csv"              # ground truth for test tickets (evaluation only)
TEST_QRELS = LABELS_DIR / "test_qrels.csv"                # (test ticket, relevant doc) pairs
TEST_HARD_NEGATIVES = LABELS_DIR / "test_hard_negatives.csv"
DUPLICATE_PAIRS = LABELS_DIR / "duplicate_pairs_to_verify.csv"
SCENARIOS_JSON = LABELS_DIR / "scenarios.json"            # evaluation only, never fed to models
TAXONOMY_PATH = DATA_DIR / "taxonomy.yaml"
USERS_PATH = DATA_DIR / "users.csv"
POLICY_PATH = ROOT / "ai" / "policy.yaml"                 # operational rules not in the dataset

PROCESSED_TICKETS = PROCESSED_DIR / "tickets.parquet"

# ----------------------------------------------------------------------------- artifacts
INDEX_DIR = ROOT / "index"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "faiss.index"
ID_MAP_PATH = INDEX_DIR / "id_map.json"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
META_DB_PATH = INDEX_DIR / "metadata.sqlite"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
TUNED_PATH = MODELS_DIR / "tuned_thresholds.json"

APP_DB_PATH = Path(os.getenv("APP_DB_PATH", DATA_DIR / "app.sqlite"))  # audit, feedback, users (backend)
LLM_CACHE_PATH = DATA_DIR / "llm_cache.sqlite"

# ----------------------------------------------------------------------------- models
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# PII engine: "presidio" (Presidio + regex) or "regex" (regex only, faster, no spaCy)
PII_ENGINE = os.getenv("PII_ENGINE", "presidio")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ----------------------------------------------------------------------------- LLM
# Grok (xAI) exposes an OpenAI-compatible API. Keys starting with "gsk_" are Groq keys,
# which are also OpenAI-compatible; the base URL and default model switch automatically.
LLM_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
_is_groq = LLM_API_KEY.startswith("gsk_")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL", "https://api.groq.com/openai/v1" if _is_groq else "https://api.x.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b" if _is_groq else "grok-4-fast-non-reasoning")
# "grok" / "groq" = call that API; "offline" = deterministic extractive generator (no API, used when no key)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", ("groq" if _is_groq else "grok") if LLM_API_KEY else "offline").lower()
LLM_TEMPERATURE = 0.0
# Reasoning models (gpt-oss, grok-*-reasoning, qwen3) spend part of max_tokens on hidden reasoning:
# keep it low and give ~800 tokens of answer plus headroom.
LLM_IS_REASONING = any(k in LLM_MODEL for k in ("gpt-oss", "qwen3", "-reasoning")) and "non-reasoning" not in LLM_MODEL
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1600" if LLM_IS_REASONING else "800"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE", "1") == "1"  # cache outputs so demos survive API outages
PROMPT_VERSION = "resolve-v1.0"
CLASSIFY_PROMPT_VERSION = "classify-assist-v1.0"

# ----------------------------------------------------------------------------- classification (FR-01)
CLS_CONF_MIN = 0.70          # top-1 probability needed to trust the model alone
CLS_MARGIN_MIN = 0.20        # top-1 minus top-2 needed to trust the model alone
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAMS = (1, 2)
VALIDATION_FRACTION = 0.15   # newest share of historical tickets used for model selection / tuning

# ----------------------------------------------------------------------------- retrieval (FR-02, FR-07)
RECALL_TOP_K = 50            # per searcher
DENSE_SEARCH_K = 200         # FAISS candidates before the access/metadata filter
RRF_K = 60
RERANK_TOP_N = 30
RERANK_TOP_N_KB = 10         # ~140 KB chunks: top 10 fused is enough to pick the best 3
RETURN_TICKETS = 5
RETURN_KB = 3
BOOST_SAME_APP = 0.10
BOOST_SHARED_ERROR = 0.15
BOOST_SAME_CATEGORY = 0.05
BOOST_RECENT = 0.05
RECENT_DAYS = 365
FEEDBACK_BOOST_PER_VOTE = 0.01
FEEDBACK_BOOST_MAX = 0.05
# Match score = sigmoid(A * cross_encoder_logit + B). A/B are fitted on labelled historical
# pairs (Platt scaling, ai.retrieve.calibrate) so the score is P(same issue); defaults = plain sigmoid.
RERANK_PLATT_A = 1.0
RERANK_PLATT_B = 0.0
RERANK_MAX_LENGTH = 256
MATCH_DUPLICATE = 0.85       # >= and same application -> likely duplicate
MATCH_SIMILAR = 0.60         # >= -> similar
MATCH_WEAK = 0.40            # >= -> weak (shown greyed out); below -> dropped
RESOLVED_STATUSES = ("Resolved", "Closed")

# ----------------------------------------------------------------------------- resolution (FR-03, FR-04)
EVIDENCE_MAX_TICKETS = 5
EVIDENCE_MAX_KB = 3
EVIDENCE_KB_MIN_SCORE = 0.60
EXCERPT_MAX_WORDS = 40
EXCERPT_FUZZY_MIN = 90
MAX_PATHS = 3
RISKY_PATTERNS = [
    r"\bdelete\b", r"\bformat\b", r"\breset all\b", r"\bdisable (?:the )?(?:security|antivirus|firewall|mfa|defender|bitlocker)\b",
    r"\bwipe\b", r"\bdrop (?:table|database)\b", r"\btruncate\b", r"\bkill(?:ed)?\b", r"\bpurge\b", r"\breimage\b",
    r"\bfactory reset\b", r"\brestor(?:e|ed) (?:the )?(?:database|backup)\b", r"\buninstall(?:ed)? (?:the )?(?:update|patch)\b",
]

# ----------------------------------------------------------------------------- confidence (FR-05)
W_CLASSIFICATION = 0.25
W_RETRIEVAL = 0.35
W_AGREEMENT = 0.20
W_VALIDITY = 0.20
BAND_HIGH = 0.75
BAND_MEDIUM = 0.55
RULE_MIN_RETRIEVAL = 0.60
RULE_MIN_VALIDITY = 0.50
AGREEMENT_TOP_N = 3


# ----------------------------------------------------------------------------- access (section 12)
def _load_users() -> list[tuple[str, str, set[str]]]:
    """users.csv (user_id, role, can_see) from the dataset defines the prototype roles."""
    if not USERS_PATH.exists():
        return [("analyst1", "L1", {"Public", "Internal"}),
                ("admin1", "L2_Admin", {"Public", "Internal", "Restricted"})]
    with open(USERS_PATH, newline="", encoding="utf-8") as f:
        return [(r["user_id"], r["role"], {s.strip() for s in r["can_see"].split(",") if s.strip()})
                for r in csv.DictReader(f)]


USERS = _load_users()
DEFAULT_USERS = [(u, r) for u, r, _ in USERS]
ROLE_SENSITIVITY = {r: s for _, r, s in USERS}
# roles that may see Restricted content also own the full audit log export
ROLES_WITH_AUDIT = {r for r, s in ROLE_SENSITIVITY.items() if "Restricted" in s}

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ----------------------------------------------------------------------------- authentication (JWT)
# JWT_SECRET signs the tokens: set a long random value in .env (python -m backend.auth --generate-secret).
# Without it a random secret is generated per server start, so tokens stop working after a restart.
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
# Initial password for the users in data/users.csv that have none yet (demo only; set per user with
# python -m backend.auth --set-password <user>). Empty = nobody can log in until a password is set.
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "")
LOGIN_MAX_FAILURES = 5         # failed logins per user within the window before a lockout
LOGIN_LOCKOUT_SECONDS = 300

# ----------------------------------------------------------------------------- tuned overrides
_TUNABLE = {"CLS_CONF_MIN", "CLS_MARGIN_MIN", "MATCH_DUPLICATE", "MATCH_SIMILAR", "MATCH_WEAK",
            "RERANK_PLATT_A", "RERANK_PLATT_B"}


def load_tuned() -> dict:
    if TUNED_PATH.exists():
        try:
            return json.loads(TUNED_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_tuned(values: dict) -> None:
    """Merge tuned values into models/tuned_thresholds.json."""
    current = load_tuned()
    current.update(values)
    TUNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")


for _k, _v in load_tuned().items():
    if _k in _TUNABLE:
        globals()[_k] = _v


def llm_settings() -> dict:
    return {"provider": LLM_PROVIDER, "model": LLM_MODEL if LLM_PROVIDER != "offline" else "offline-extractive",
            "base_url": LLM_BASE_URL if LLM_PROVIDER != "offline" else None}
