import re
import sqlite3
import uuid
from datetime import datetime

import pandas as pd
import pytest

from ai import access, config
from backend import audit
from ai.preprocess import preprocess


def test_l1_cannot_see_restricted_docs():
    doc = {"doc_type": "ticket", "sensitivity": "Restricted", "root_cause": "x", "approved": True}
    assert access.redact_doc(doc, "L1") is None
    assert access.redact_doc(doc, "L2_Admin")["root_cause"] == "x"


def test_restricted_fields_are_redacted_for_l1():
    doc = {"doc_type": "ticket", "sensitivity": "Internal", "root_cause": "brute force from IP range",
           "resolution_notes": "revoked sessions", "resolution_text": "...", "has_resolution": True,
           "restricted_fields": "root_cause;resolution_notes"}
    shown = access.redact_doc(doc, "L1")
    assert shown["root_cause"] == access.RESTRICTED and shown["has_resolution"] is False
    assert access.redact_doc(doc, "L2_Admin")["root_cause"] == "brute force from IP range"


def test_unapproved_kb_never_visible():
    docs = pd.DataFrame([{"doc_type": "kb", "approved": False, "sensitivity": "Internal"},
                         {"doc_type": "kb", "approved": True, "sensitivity": "Internal"},
                         {"doc_type": "ticket", "approved": True, "sensitivity": "Restricted"}])
    assert access.allowed_mask(docs, "L2_Admin").tolist() == [False, True, True]
    assert access.allowed_mask(docs, "L1").tolist() == [False, True, False]


def _analysis(uid="analyst1", input_hash="h1"):
    rec = preprocess({"title": "VPN down", "description": "mail me at john.doe@corp.example.com, IP 10.1.2.3"})
    aid = str(uuid.uuid4())
    audit.write_analysis({"analysis_id": aid, "created_at": datetime.now(), "user_id": uid, "user_role": "L1",
                          "input_hash": input_hash, "input_masked": {"title": rec["title"], "description": rec["description"]},
                          "classification_json": {}, "retrieved_json": [], "evidence_ids": [],
                          "recommendation_json": {"paths": []}, "validation_json": {}, "confidence_json": {},
                          "fallback_json": None, "band": "Low", "versions": {}, "latency_ms": {}})
    return aid


def test_audit_feedback_hiding_boost_and_no_pii_in_db():
    audit.init_db()
    assert audit.get_role("analyst1") == "L1" and audit.get_role("admin1") == "L2_Admin"
    aid = _analysis()
    audit.add_feedback(aid, "match", "INC0010455", "Incorrect", "analyst1", "L1", comment="different cause")
    audit.add_feedback(aid, "match", "INC0011872", "useful", "analyst1", "L1")
    assert audit.hidden_matches("h1") == {"INC0010455"}
    adj = audit.feedback_adjustments()
    assert adj["INC0010455"] < 0 < adj["INC0011872"] <= config.FEEDBACK_BOOST_MAX
    audit.set_decision(aid, "accepted", "analyst1", "L1")
    assert audit.get_analysis(aid)["final_decision"] == "Accepted"
    with pytest.raises(PermissionError):
        audit.add_feedback(aid, "match", "INC1", "useful", "someone_else", "L1")
    with pytest.raises(ValueError):
        audit.add_feedback(aid, "match", "INC1", "great", "analyst1", "L1")
    # PII leak test: no email / IPv4 pattern anywhere in the audit database
    with sqlite3.connect(config.APP_DB_PATH) as conn:
        text = " ".join(str(v) for row in conn.execute("SELECT * FROM analyses") for v in row)
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)


@pytest.mark.skipif(not config.FAISS_PATH.exists(), reason="index not built")
def test_retrieval_never_returns_restricted_docs_to_l1():
    from ai.embeddings import embed_queries
    from ai.retrieve import Retriever
    r = Retriever()
    # the phishing scenario's history is entirely Restricted in the dataset
    q = preprocess({"title": "Clicked phishing email and typed credentials",
                    "description": "I typed my password on a login page that looked like OneDrive.",
                    "application": "OneDrive"})
    e = embed_queries([q["search_text"]])[0]
    l1 = r.search(q, e, "L1", top_k=10, keep_weak=True)
    admin = r.search(q, e, "L2_Admin", top_k=10, keep_weak=True)
    assert all(r.docs_by_id[m["doc_id"]]["sensitivity"] != "Restricted" for m in l1)
    assert any(r.docs_by_id[m["doc_id"]]["sensitivity"] == "Restricted" for m in admin)


@pytest.mark.skipif(not config.FAISS_PATH.exists(), reason="index not built")
def test_filters_are_applied_before_search():
    from ai.embeddings import embed_queries
    from ai.retrieve import Retriever
    r = Retriever()
    q = preprocess({"title": "cannot connect to vpn", "description": "error 809"})
    e = embed_queries([q["search_text"]])[0]
    res = r.search(q, e, "L2_Admin", filters={"application": ["Outlook"], "status": ["Closed"]}, top_k=10, keep_weak=True)
    assert all(m["application"] == "Outlook" and m["status"] == "Closed" for m in res)
