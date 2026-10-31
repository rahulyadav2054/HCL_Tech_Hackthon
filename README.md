# IT Ticket Resolution Assistant

Analyzes one incoming IT ticket and returns:

1. a **classification** (category, subcategory, priority) with calibrated confidence and an explanation,
2. **similar historical tickets** with a score, a *likely duplicate / similar / weak* label and the reasons for each match,
3. **one to three resolution paths**, every one backed by **verified citations** (source ID, exact excerpt, why it applies),
4. a **confidence verdict** (High / Medium / Low / Escalate) that routes weak, contradictory or critical cases to a human.

It **analyzes and recommends only**: it never closes tickets, runs remediation or writes to an ITSM tool. Every output
points to a ticket field or an approved KB article. Classic ML classifies, hybrid search finds the evidence, the LLM
writes only from that evidence, and code verifies every citation.

Built for the HCLTech *Future Ready AI Engineer Hackathon* use case "Intelligent IT Ticket Analysis and Resolution
Assistant", on the **it-ticket-dataset** (synthetic; see `data/DATA_CARD.md`).

---

## Setup and run (Windows PowerShell)

Requirements: Python 3.11, ~2 GB disk for models, CPU only (8 GB RAM: run one model-loading process at a time).

```powershell
cd C:\Users\Hp\Desktop\Project1\it-ticket-assistant
python -m venv .venv
.venv\Scripts\Activate.ps1             # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` (never commit it) with the LLM key, the JWT secret and a demo password:

```powershell
python -m backend.auth --generate-secret      # prints JWT_SECRET=... : copy that line into .env
```

```text
GROK_API_KEY=gsk_...            # Groq key (gsk_) or xAI Grok key (xai-); optional
JWT_SECRET=<the generated value>
DEMO_PASSWORD=<a password for analyst1, analyst2 and admin1>
```

Build once (about 8 minutes), then run the backend and the frontend in two terminals:

```powershell
python -m ai.build_index               # preprocess + BM25/FAISS index + reranker calibration
python -m ai.classify --train          # 3 calibrated classifiers

uvicorn backend.main:app --port 8000   # terminal 1: API, docs at http://localhost:8000/docs
streamlit run frontend/app.py          # terminal 2: UI at http://localhost:8501
```

Log in on the UI as `analyst1` / `analyst2` (role L1) or `admin1` (role L2_Admin) with the demo password.
Change a password any time with `python -m backend.auth --set-password analyst1`.

Optional: `python -m ai.llm --check` (LLM key and model), `python -m pytest -q` (unit tests),
`python -m ai.eval.run_all` (evaluation report), `python -m ai.dataset --import <zip>` (load a new dataset version).

### LLM

* Called through the OpenAI-compatible API with the `openai` SDK. A `gsk_` key is detected as **Groq**
  (`https://api.groq.com/openai/v1`, default model `openai/gpt-oss-120b`, low reasoning effort); an `xai-` key uses
  **Grok** (`https://api.x.ai/v1`). Override with `LLM_MODEL` / `LLM_BASE_URL` in `.env`.
* Only the PII-masked ticket and the evidence pack (max 5 tickets + 3 KB sections) are sent.
* No key, or the API fails: a deterministic **offline extractive generator** builds the paths from the evidence text.
  The generator used is recorded in every response and audit row. Outputs are cached in `data/llm_cache.sqlite`.

---

## Architecture

```
frontend/  Streamlit UI ──HTTP + Bearer JWT──▶ backend/  FastAPI ──analyze(ticket, role)──▶ ai/  analysis pipeline
                                                │ auth.py (JWT), audit.py (SQLite)          │ index/, models/, data/
                                                ▼                                          ▼
                                           data/app.sqlite                          Groq API (masked text only)
```

| Folder | Contents |
|---|---|
| `ai/` | `config.py`, `policy.yaml`, `dataset.py`, `preprocess.py`, `embeddings.py`, `build_index.py`, `classify.py`, `retrieve.py`, `llm.py`, `resolve.py`, `validate.py`, `confidence.py`, `access.py`, `analyzer.py`, `eval/` |
| `backend/` | `main.py` (endpoints), `schemas.py`, `auth.py` (JWT + passwords), `service.py` (role, analyzer, audit write), `audit.py` (SQLite) |
| `frontend/` | `app.py` (Streamlit: login, Analyze, Search, Audit log, Metrics) |
| `data/` | `raw/`, `labels/`, `taxonomy.yaml`, `users.csv`, `DATA_CARD.md`, `generator/` |
| `index/`, `models/` | built by the setup commands |
| `reports/`, `tests/` | evaluation output, unit tests |

| Requirement | Module |
|---|---|
| FR-01 Classification | `ai/classify.py`: calibrated logistic regression (TF-IDF, embedding or combined, chosen on validation), subcategory restricted to the taxonomy, priority rules, LLM second opinion on close calls |
| FR-02 Similarity | `ai/retrieve.py`: role filter, BM25 top 50 + dense top 50, RRF, boosts, cross-encoder rerank of 30, Platt-calibrated match score |
| FR-03 Resolution | `ai/resolve.py`, `ai/llm.py`: evidence pack, grounded prompt, strict JSON, one retry |
| FR-04 Traceability | `ai/validate.py`: cited IDs must be in the pack, excerpts must match the source (exact or rapidfuzz >= 90) |
| FR-05 Confidence + fallback | `ai/confidence.py`: 0.25 C + 0.35 R + 0.20 A + 0.20 V, hard rules, bands, routing, clarifying questions |
| FR-06 Feedback | `backend/audit.py`, `POST /feedback` |
| FR-07 Search + filters | `POST /search` |
| FR-08 Audit | `backend/audit.py`, `GET /audit/export` |
| Access control | `ai/access.py` (Restricted filtered before retrieval, redaction), `backend/auth.py` (JWT) |

---

## Authentication (JWT)

* `POST /auth/login` with `{"username", "password"}` returns `{"access_token", "token_type": "bearer", "expires_in", "user_id", "role"}`.
* Every other endpoint except `/health` needs `Authorization: Bearer <access_token>`; missing, invalid, tampered or
  expired tokens get `401`.
* Tokens are HS256, signed with `JWT_SECRET`, valid `JWT_EXPIRE_MINUTES` (default 60), with `sub`, `role`, `iat`,
  `nbf`, `exp`, `iss`, `jti` claims. Without `JWT_SECRET` a random secret is used per server start.
* The role is **read from the database on every request**; the role inside the token is informational.
* Passwords are stored as salted PBKDF2-SHA256 hashes (310,000 iterations), never in plain text. Users come from
  `data/users.csv`; initial passwords from `DEMO_PASSWORD`, changed with `python -m backend.auth --set-password <user>`.
* After 5 failed logins for a username within 5 minutes, logins for it return `429` until the window passes.
* Logout is client-side (the UI drops the token); tokens are not revocable before they expire.

## API

Swagger UI at http://localhost:8000/docs: call `POST /auth/login`, click **Authorize**, paste the token.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/auth/login` | Username + password -> JWT |
| GET | `/auth/me` | User and current role for the token |
| POST | `/analyze` | Full pipeline on one ticket (FR-01..05, FR-08) |
| POST | `/search` | Free-text search with filters, no LLM (FR-02, FR-07) |
| GET | `/tickets/{id}` | Historical ticket or KB article, role-checked and redacted (FR-04) |
| POST | `/feedback` | useful / incorrect / unresolved (FR-06) |
| PATCH | `/analyses/{id}/decision` | Accepted / Edited / Rejected / Escalated (FR-08) |
| GET | `/audit/export?from=&to=&format=csv\|json` | L2_Admin exports all analyses, L1 only their own |
| GET | `/health` | Versions, thresholds, filter catalog, aggregate metrics (public) |

```powershell
$login = Invoke-RestMethod -Method Post http://localhost:8000/auth/login -ContentType "application/json" `
  -Body '{"username":"analyst1","password":"<password>"}'
$h = @{ Authorization = "Bearer $($login.access_token)" }
Invoke-RestMethod -Method Post http://localhost:8000/analyze -Headers $h -ContentType "application/json" `
  -Body '{"title":"VPN not connecting after password change","description":"Error 809 on GlobalProtect from home since I changed my password.","application":"GlobalProtect VPN","environment":"Production"}'
```

## UI

| Page | What it does |
|---|---|
| Login | Username + password; the token is kept in the browser session only |
| Analyze ticket | Sample tickets, confidence banner, classification tiles with explanation and label feedback, cited resolution paths (click a source to open it with the excerpt highlighted), similar tickets with feedback, fallback panel, decision bar |
| Search | Query + filters (application, category, status, group, environment, dates) |
| Audit log | L2_Admin only: analyses table, full JSON, CSV export |
| Metrics | Live feedback metrics + offline evaluation charts |

Access-control demo: analyze the "Security + access control (phishing)" sample as analyst1, then as admin1. The phishing
history is Restricted: L1 gets no matches from it, the admin sees it as evidence; both escalate to Security-Ops.

---

## Evaluation

`python -m ai.eval.run_all` runs every script below (one at a time) and writes `reports/evaluation_report.md`.

| Dimension | Script | Ground truth |
|---|---|---|
| Classification | `ai/eval/eval_classification.py` | `test_labels.csv` (400 standard tickets; by test type) |
| Duplicate retrieval | `ai/eval/eval_retrieval.py --tune` | `test_qrels.csv`, `test_hard_negatives.csv` |
| Resolution usefulness | `ai/eval/eval_resolution.py` | qrels + `reference_resolution`; blind rating sheets in `data/labels/ratings/` |
| Confidence + fallback | `ai/eval/eval_confidence.py` | `expected_behavior`, `expected_assignment_group` |
| Traceability, access, PII | `ai/eval/eval_traceability.py` | audit log scan, L1 vs L2_Admin runs, prompt injection |
| Latency | `ai/eval/eval_latency.py` | p50 / p95 per stage |
| Worked examples | `ai/eval/worked_examples.py` | `reports/worked_examples.md` + audit JSON |

Evaluations call the AI and backend layers directly (no HTTP, no login) and write to `data/eval_audit.sqlite`.

## Assumptions, trade-offs, limitations

* Data is synthetic and template-based (the data card says real data would score lower); English only.
* Calibrated linear models over a fine-tuned transformer: fast, explainable, calibrated.
* Reranker score is Platt-calibrated on labelled historical pairs so the 0.85 / 0.60 / 0.40 bands mean P(same issue).
* Strict citation validation drops some valid paraphrases; hand-set confidence weights are kept for transparency.
* Prototype auth: no SSO, no token revocation, lockout is in memory (resets on restart).
