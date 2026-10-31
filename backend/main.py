"""FastAPI backend: the seven endpoints of section 13. Swagger UI at /docs.

Authentication: POST /auth/login returns a JWT; every other endpoint except /health needs
`Authorization: Bearer <token>` (backend/auth.py). Users and roles come from data/users.csv.
Run:  uvicorn backend.main:app --port 8000
"""
from __future__ import annotations

import csv
import io
import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from ai import access
from backend import audit, auth
from backend.auth import current_user
from backend.schemas import AnalyzeRequest, DecisionRequest, FeedbackRequest, LoginRequest, SearchRequest, TokenResponse
from backend.service import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()  # load index, models and warm the reranker once at startup
    yield


app = FastAPI(
    title="IT Ticket Resolution Assistant",
    description="Analyzes an incoming IT ticket: classification with confidence, similar historical tickets, "
                "evidence-backed resolution paths with verified citations, and a confidence verdict that routes "
                "weak cases to a human. Analyze and recommend only: it never closes tickets or changes systems.\n\n"
                "**Authentication:** call `POST /auth/login` with your username and password, then click "
                "**Authorize** and paste the `access_token`. Users: analyst1, analyst2 (role L1), admin1 (role L2_Admin).",
    version="2.1.0",
    lifespan=lifespan,
)


@app.post("/auth/login", tags=["auth"], response_model=TokenResponse, summary="Log in and get a JWT")
def login(req: LoginRequest):
    username = req.username.strip().lower()  # "Analyst1 " -> "analyst1"; the password is used exactly as typed
    wait = auth.THROTTLE.locked_for(username)
    if wait:
        raise HTTPException(429, f"too many failed logins: try again in {wait} s", headers={"Retry-After": str(wait)})
    role = auth.authenticate(username, req.password)
    if role is None:
        auth.THROTTLE.failed(username)
        raise HTTPException(401, "invalid username or password", headers={"WWW-Authenticate": "Bearer"})
    auth.THROTTLE.succeeded(username)
    token, ttl = auth.create_access_token(username, role)
    return TokenResponse(access_token=token, expires_in=ttl, user_id=username, role=role)


@app.get("/auth/me", tags=["auth"], summary="Who the token belongs to")
def me(user: dict = Depends(current_user)):
    return user


@app.post("/analyze", tags=["analysis"], summary="Full pipeline on one ticket (FR-01..05, FR-08)")
def analyze(req: AnalyzeRequest, user: dict = Depends(current_user)):
    ticket = {"title": req.title, "description": req.description, "application": req.application,
              "environment": req.environment, "filters": req.filters.as_dict() if req.filters else {}}
    return get_service().analyze(ticket, user["user_id"], use_llm=req.use_llm)


@app.post("/search", tags=["search"], summary="Free-text search with filters, no LLM (FR-02, FR-07)")
def search(req: SearchRequest, user: dict = Depends(current_user)):
    return get_service().search(req.query, user["user_id"], req.filters.as_dict() if req.filters else {},
                                req.top_k, req.include_kb)


@app.get("/tickets/{ticket_id}", tags=["search"], summary="Full historical ticket or KB article (role-checked, FR-04)")
def get_ticket(ticket_id: str, user: dict = Depends(current_user)):
    r = get_service().retriever
    tid = ticket_id.split("§")[0].upper()
    if tid.startswith("KB"):
        sections = [d for d in r.docs_by_id.values() if d["doc_type"] == "kb" and d["source_id"] == tid]
        if not sections:
            raise HTTPException(404, f"{tid} not found or not approved")
        visible = [access.redact_doc(d, user["role"]) for d in sections]
        if not any(visible):
            raise HTTPException(403, "not visible for your role")
        first = sections[0]
        return {"id": tid, "type": "kb", "title": first["title"], "application": first["applies_to"],
                "category": first["category"], "subcategory": first["subcategory"], "owner_group": first["assignment_group"],
                "last_reviewed": first["created_at"], "approved": True,
                "sections": {d["section"]: d["description"] for d in visible if d}}
    doc = r.docs_by_id.get(tid)
    if doc is None:
        raise HTTPException(404, f"{tid} is not a historical ticket in the index")
    shown = access.redact_doc(doc, user["role"])
    if shown is None:
        raise HTTPException(403, f"{tid} is {doc['sensitivity']} and not visible for role {user['role']}")
    keys = ("title", "description", "application", "environment", "category", "subcategory", "priority",
            "assignment_group", "status", "created_at", "resolved_at", "root_cause", "resolution_notes",
            "kb_refs", "sensitivity")
    return {"id": tid, "type": "ticket", **{k: shown.get(k) for k in keys},
            "redacted_fields": shown.get("redacted_fields", [])}


@app.post("/feedback", tags=["feedback"], summary="Store an analyst verdict (FR-06)")
def feedback(req: FeedbackRequest, user: dict = Depends(current_user)):
    try:
        return audit.add_feedback(req.analysis_id, req.target_type, req.target_id, req.verdict, user["user_id"],
                                  user["role"], req.corrected_label, req.comment)
    except KeyError:
        raise HTTPException(404, "analysis not found")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.patch("/analyses/{analysis_id}/decision", tags=["feedback"],
           summary="Record Accepted / Edited / Rejected / Escalated (FR-08)")
def decision(analysis_id: str, req: DecisionRequest, user: dict = Depends(current_user)):
    try:
        return audit.set_decision(analysis_id, req.decision, user["user_id"], user["role"], req.comment or "")
    except KeyError:
        raise HTTPException(404, "analysis not found")
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/audit/export", tags=["audit"], summary="CSV/JSON of analyses + feedback (FR-08)")
def export(date_from: str | None = Query(None, alias="from", examples=["2026-10-01"]),
           date_to: str | None = Query(None, alias="to"),
           format: Literal["json", "csv"] = "json", user: dict = Depends(current_user)):
    """L2_Admin exports every analysis; L1 exports only their own. The role filter and masking are
    re-applied: references to documents the role cannot see are removed; text is already PII-masked."""
    owner = None if access.can_export(user["role"]) else user["user_id"]
    try:
        rows = audit.list_analyses(date_from, date_to, owner)
    except ValueError:
        raise HTTPException(422, "from/to must be ISO dates (YYYY-MM-DD)")
    docs = get_service().retriever.docs_by_id
    rows = [access.redact_analysis(r, user["role"], docs) for r in rows]
    fb = audit.list_feedback([r["analysis_id"] for r in rows])
    if format == "json":
        return JSONResponse({"role": user["role"], "scope": "all" if owner is None else "own",
                             "analyses": rows, "feedback": fb})
    by_analysis: dict[str, list] = {}
    for f in fb:
        by_analysis.setdefault(f["analysis_id"], []).append(f)
    buf = io.StringIO()
    cols = list(rows[0].keys()) + ["feedback_json"] if rows else ["analysis_id"]
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({**{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()},
                    "feedback_json": json.dumps(by_analysis.get(r["analysis_id"], []))})
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=audit_export.csv"})


@app.get("/health", tags=["ops"], summary="Model and index versions")
def health():
    return get_service().health()
