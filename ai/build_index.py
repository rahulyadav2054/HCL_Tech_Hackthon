"""Index builder (Phase 3): BM25 + FAISS over historical tickets and approved KB sections.

Both indexes are keyed by the same doc_id. Run once offline:
    python -m ai.build_index            (runs preprocessing first if data/processed is stale)

Document set
  * one document per historical ticket (data/raw/historical_tickets.csv). Resolved/Closed tickets with
    resolution notes carry has_resolution = true; Open / In Progress tickets are indexed for duplicate
    detection only.
  * one document per section (Symptoms, Cause, Steps) of every APPROVED KB article, so citations can
    point at a section: doc_id = "KB0042§Steps". Drafts / superseded articles are never indexed.
scenario_id is never indexed (evaluation only).

Outputs: index/bm25.pkl, index/faiss.index, index/embeddings.npy, index/id_map.json,
         index/metadata.sqlite (documents table), index/manifest.json
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ai import config, dataset
from ai.embeddings import embed_docs
from ai.preprocess import build_search_text, extract_entities, load_processed, mask_pii, normalize, process_all

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_PLACEHOLDER = re.compile(r"<(?:email|phone|ip|user|host)>")
KB_SECTIONS = (("Symptoms", "symptoms"), ("Cause", "cause"), ("Steps", "resolution_steps"))


def tokenize(text: str) -> list[str]:
    """Lowercase, keep error codes intact (ora-12541, 0x80070005, 5.2.2), drop stop words."""
    text = _PLACEHOLDER.sub(" ", text.lower())
    return [t for t in _TOKEN.findall(text) if (t not in ENGLISH_STOP_WORDS and len(t) > 1) or t.isdigit()]


def file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def ensure_processed() -> pd.DataFrame:
    sources = (config.HISTORICAL_TICKETS, config.TEST_TICKETS)
    stale = (not config.PROCESSED_TICKETS.exists()
             or config.PROCESSED_TICKETS.stat().st_mtime < max(p.stat().st_mtime for p in sources))
    if stale:
        print("processed data missing or stale -> running preprocessing")
        process_all()
    return load_processed("historical")


def ticket_documents(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["doc_id"] = t["ticket_id"]
    t["doc_type"] = "ticket"
    t["source_id"] = t["ticket_id"]
    t["section"] = ""
    t["applies_to"] = t["application"]
    t["text"] = t["title"] + ". " + t["description"]
    t["has_resolution"] = t["status"].isin(config.RESOLVED_STATUSES) & (t["resolution_notes"].str.len() > 0)
    t["approved"] = True
    return t


def kb_documents() -> pd.DataFrame:
    rows = []
    for a in dataset.kb_articles(approved_only=True).to_dict("records"):
        apps = dataset.split_list(a["applies_to"])
        kb_codes = extract_entities(a["error_codes"])["error_codes"] if a["error_codes"] else []
        for section, col in KB_SECTIONS:
            content, _ = mask_pii(normalize(a[col].replace("\n", " ")))
            text = f"{a['title']}. {section}: {content}"
            ent = extract_entities(text)
            ent["error_codes"] = list(dict.fromkeys(ent["error_codes"] + kb_codes))
            rows.append({
                "doc_id": f"{a['kb_id']}§{section}", "doc_type": "kb", "source_id": a["kb_id"], "section": section,
                "ticket_id": None, "title": a["title"], "description": content, "text": text,
                "search_text": build_search_text(a["title"], content, ent), "entities": ent,
                "root_cause": a["cause"] if section == "Cause" else "", "resolution_notes": "",
                "resolution_text": text, "has_resolution": True, "approved": True,
                "application": apps[0] if apps else "", "applies_to": ", ".join(apps),
                "category": a["category"], "subcategory": a["subcategory"], "priority": None,
                "environment": "Production", "status": "Approved", "assignment_group": a["owner_group"],
                "created_at": a["last_reviewed"], "resolved_at": a["last_reviewed"],
                "sensitivity": "Internal", "restricted_fields": "", "kb_refs": "",
            })
    return pd.DataFrame(rows)


DOC_COLUMNS = ["doc_id", "doc_type", "source_id", "section", "title", "description", "text", "search_text",
               "entities", "root_cause", "resolution_notes", "resolution_text", "has_resolution", "approved",
               "application", "applies_to", "category", "subcategory", "priority", "environment", "status",
               "assignment_group", "created_at", "resolved_at", "sensitivity", "restricted_fields", "kb_refs"]


def build():
    t0 = time.time()
    df = ensure_processed()
    docs = pd.concat([ticket_documents(df), kb_documents()], ignore_index=True)[DOC_COLUMNS]
    docs["has_resolution"] = docs["has_resolution"].astype(bool)
    docs["approved"] = docs["approved"].astype(bool)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    doc_ids = docs["doc_id"].tolist()

    # BM25
    bm25 = BM25Okapi([tokenize(t) for t in docs["search_text"]])
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids}, f)

    # Dense (bge-small, normalized -> inner product == cosine)
    print(f"embedding {len(docs)} documents with {config.EMBED_MODEL} ...")
    emb = embed_docs(docs["search_text"].tolist(), show_progress=False)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(config.FAISS_PATH))
    np.save(config.EMBEDDINGS_PATH, emb)
    config.ID_MAP_PATH.write_text(json.dumps(doc_ids), encoding="utf-8")

    # Metadata table (SQLite) used for filtering
    meta = docs.copy()
    meta["entities"] = meta["entities"].apply(json.dumps)
    if config.META_DB_PATH.exists():
        config.META_DB_PATH.unlink()
    with sqlite3.connect(config.META_DB_PATH) as conn:
        meta.to_sql("documents", conn, index=False)
        conn.execute("CREATE INDEX idx_doc ON documents(doc_id)")

    data_hash = file_hash(config.HISTORICAL_TICKETS, config.KB_ARTICLES)
    manifest = {
        "index_version": data_hash[:12],
        "dataset": "it-ticket-dataset (synthetic, see data/DATA_CARD.md)",
        "embedding_model": config.EMBED_MODEL,
        "embedding_dim": int(emb.shape[1]),
        "faiss_index": "IndexFlatIP",
        "doc_count": len(docs),
        "ticket_docs": int((docs.doc_type == "ticket").sum()),
        "tickets_with_resolution": int(((docs.doc_type == "ticket") & docs.has_resolution).sum()),
        "kb_articles": int(docs.loc[docs.doc_type == "kb", "source_id"].nunique()),
        "kb_chunks": int((docs.doc_type == "kb").sum()),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "data_hash": data_hash,
        "build_seconds": round(time.time() - t0, 1),
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    # Platt-calibrate the reranker on labelled historical pairs (match score = P(same issue))
    from ai.retrieve import calibrate
    manifest["reranker_calibration"] = calibrate()
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    build()
