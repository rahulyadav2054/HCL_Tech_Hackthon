"""Role-based visibility and redaction (section 12).

Restricted content is removed at three points:
  1. before retrieval  -> allowed_mask() is applied before BM25/FAISS results are fused
  2. before the LLM    -> redact_doc() runs on every evidence item
  3. before display / export -> redact_doc() / redact_analysis() on API output

Roles come from data/users.csv (user_id, role, can_see): L1 (analyst1, analyst2) sees Public + Internal
tickets and approved KB; L2_Admin (admin1) also sees Restricted and exports the full audit log.
Field-level rule: a ticket may list restricted_fields (e.g. root_cause of a security case);
those fields are shown as "[restricted]" to roles that cannot see Restricted content.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai import config

RESTRICTED = "[restricted]"


def allowed_sensitivities(role: str) -> set[str]:
    return config.ROLE_SENSITIVITY.get(role, {"Public"})


def can_see_restricted(role: str) -> bool:
    return "Restricted" in allowed_sensitivities(role)


def can_export(role: str) -> bool:
    return role in config.ROLES_WITH_AUDIT


def allowed_mask(docs: pd.DataFrame, role: str) -> np.ndarray:
    """Boolean mask over the document table: visible to this role and retrievable."""
    vis = docs["sensitivity"].isin(allowed_sensitivities(role)).to_numpy()
    kb_ok = (docs["doc_type"] != "kb").to_numpy() | docs["approved"].astype(bool).to_numpy()
    return vis & kb_ok


def allowed_ids(docs: pd.DataFrame, role: str) -> set[str]:
    return set(docs.loc[allowed_mask(docs, role), "doc_id"])


def can_view(doc: dict, role: str) -> bool:
    if doc.get("doc_type") == "kb" and not doc.get("approved", True):
        return False
    return doc.get("sensitivity", "Internal") in allowed_sensitivities(role)


def restricted_fields(doc: dict) -> list[str]:
    raw = doc.get("restricted_fields") or ""
    return [f for f in str(raw).split(";") if f]


def redact_doc(doc: dict, role: str) -> dict | None:
    """Return a copy safe for this role, or None if the role may not see the document at all."""
    if not can_view(doc, role):
        return None
    out = dict(doc)
    if not can_see_restricted(role):
        fields = restricted_fields(doc)
        for f in fields:
            if f in out:
                out[f] = RESTRICTED
        if fields:
            out["resolution_text"] = RESTRICTED
            out["has_resolution"] = False  # nothing usable as evidence for this role
            out["redacted_fields"] = fields
    return out


def visible_doc_ids(doc_ids, docs_by_id: dict, role: str) -> set[str]:
    return {d for d in doc_ids if d in docs_by_id and can_view(docs_by_id[d], role)}


def redact_analysis(record: dict, role: str, docs_by_id: dict) -> dict:
    """Re-apply the role filter to a stored analysis before export (section 11.3).

    Removes references to documents the role cannot see from retrieved candidates, evidence IDs
    and recommendation citations. PII is already masked in the stored record.
    """
    out = dict(record)

    def ok(doc_id: str) -> bool:
        base = doc_id.split("§")[0]
        doc = docs_by_id.get(doc_id) or docs_by_id.get(base)
        return doc is None or can_view(doc, role)

    if isinstance(out.get("retrieved_json"), list):
        out["retrieved_json"] = [c for c in out["retrieved_json"] if ok(c.get("doc_id", ""))]
    if isinstance(out.get("evidence_ids"), list):
        out["evidence_ids"] = [d for d in out["evidence_ids"] if ok(d)]
    rec = out.get("recommendation_json")
    if isinstance(rec, dict):
        rec = dict(rec)
        paths = []
        for p in rec.get("paths", []):
            p = dict(p)
            p["evidence"] = [e for e in p.get("evidence", []) if ok(e.get("source_id", ""))]
            if p["evidence"]:
                paths.append(p)
        rec["paths"] = paths
        out["recommendation_json"] = rec
    return out
