"""Similar-ticket retrieval, filters and reranking (Phase 5, FR-02 / FR-07).

Stages: (1) pre-filter by role + analyst filters  (2) wide recall: BM25 top 50 + dense top 50
(3) Reciprocal Rank Fusion  (4) small explainable metadata boosts  (5) cross-encoder rerank of
the top 30 -> sigmoid match score  (6) label  (7) explain each match.

Filtering happens BEFORE search, so restricted text never reaches the reranker or the LLM.
The same search() powers /search (FR-07): free text + filters, no classification or LLM.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import threading
from datetime import datetime, timedelta
from functools import cached_property

import faiss
import numpy as np
import pandas as pd

from ai import access, config, dataset
from ai.build_index import tokenize

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
TICKET_ONLY_FILTERS = ("status", "assignment_group", "environment", "date_from", "date_to")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def match_label(score: float, same_app: bool) -> str:
    if score >= config.MATCH_DUPLICATE and same_app:
        return "likely_duplicate"
    if score >= config.MATCH_SIMILAR:
        return "similar"
    if score >= config.MATCH_WEAK:
        return "weak"
    return "dropped"


def _as_list(v):
    if v is None or v == "" or v == []:
        return None
    return [v] if isinstance(v, str) else list(v)


class Retriever:
    def __init__(self):
        with open(config.BM25_PATH, "rb") as f:
            b = pickle.load(f)
        self.bm25, bm25_ids = b["bm25"], b["doc_ids"]
        self.index = faiss.read_index(str(config.FAISS_PATH))
        self.emb = np.load(config.EMBEDDINGS_PATH)
        self.doc_ids = json.loads(config.ID_MAP_PATH.read_text(encoding="utf-8"))
        if bm25_ids != self.doc_ids:
            raise RuntimeError("BM25 and FAISS id maps differ; rebuild the index")
        with sqlite3.connect(config.META_DB_PATH) as conn:
            docs = pd.read_sql("SELECT * FROM documents", conn)
        docs = docs.set_index("doc_id").loc[self.doc_ids].reset_index()
        docs["entities"] = docs["entities"].apply(json.loads)
        docs["has_resolution"] = docs["has_resolution"].astype(bool)
        docs["approved"] = docs["approved"].astype(bool)
        docs["created_dt"] = pd.to_datetime(docs["created_at"], errors="coerce")
        docs["resolved_dt"] = pd.to_datetime(docs["resolved_at"].replace("", None), errors="coerce")
        self.docs = docs
        self.docs_by_id = {r["doc_id"]: r for r in docs.drop(columns=["created_dt", "resolved_dt"]).to_dict("records")}
        self.row_of = {d: i for i, d in enumerate(self.doc_ids)}
        self.manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
        self._ce_lock = threading.Lock()

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=config.RERANK_MAX_LENGTH)

    # ------------------------------------------------------------------ 1. pre-filter
    def filter_mask(self, role: str, filters: dict | None = None, doc_type: str | None = None) -> np.ndarray:
        """Security filter first, then analyst filters (FR-07). Returns a boolean row mask."""
        d = self.docs
        mask = access.allowed_mask(d, role)
        if doc_type:
            mask &= (d["doc_type"] == doc_type).to_numpy()
        f = filters or {}
        is_ticket = (d["doc_type"] == "ticket").to_numpy()
        for col in ("application", "category", "status", "assignment_group", "environment"):
            vals = _as_list(f.get(col))
            if vals:
                hit = d[col].isin(vals).to_numpy()
                mask &= (hit | ~is_ticket) if col in TICKET_ONLY_FILTERS else hit
        if f.get("date_from"):
            mask &= ((d["created_dt"] >= pd.to_datetime(f["date_from"])).to_numpy() | ~is_ticket)
        if f.get("date_to"):
            mask &= ((d["created_dt"] <= pd.to_datetime(f["date_to"]) + timedelta(days=1)).to_numpy() | ~is_ticket)
        return mask

    # ------------------------------------------------------------------ 2-3. recall + fuse
    def _bm25_ranks(self, query_text: str, mask: np.ndarray, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query_text))
        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def _dense_ranks(self, q_emb: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
        n_search = min(config.DENSE_SEARCH_K, len(self.doc_ids))
        _, idx = self.index.search(q_emb.reshape(1, -1).astype("float32"), n_search)
        rows = [int(i) for i in idx[0] if i >= 0 and mask[i]][:k]
        if len(rows) < k and mask.sum() > len(rows):
            # filters removed most FAISS hits: exact search over the allowed subset instead
            allowed = np.flatnonzero(mask)
            sims = self.emb[allowed] @ q_emb.reshape(-1)
            rows = [int(allowed[i]) for i in np.argsort(-sims)[:k]]
        return rows

    def recall(self, query_text: str, q_emb: np.ndarray, mask: np.ndarray, mode: str = "hybrid") -> dict:
        k = config.RECALL_TOP_K
        bm = self._bm25_ranks(query_text, mask, k) if mode != "dense" else []
        de = self._dense_ranks(q_emb, mask, k) if mode != "bm25" else []
        cands: dict[int, dict] = {}
        for rank, row in enumerate(bm, start=1):
            cands.setdefault(row, {"row": row})["bm25_rank"] = rank
        for rank, row in enumerate(de, start=1):
            cands.setdefault(row, {"row": row})["dense_rank"] = rank
        for c in cands.values():
            c["rrf"] = sum(1.0 / (config.RRF_K + c[r]) for r in ("bm25_rank", "dense_rank") if r in c)
            c.setdefault("bm25_rank", None)
            c.setdefault("dense_rank", None)
        return cands

    # ------------------------------------------------------------------ 4. boosts
    def boost(self, cands: dict, q_entities: dict, q_app: str, category: str | None,
              feedback_adj: dict | None = None, now: datetime | None = None) -> None:
        now = now or datetime.now()
        recent_cut = now - timedelta(days=config.RECENT_DAYS)
        q_codes = set(q_entities.get("error_codes", []))
        q_apps = ({q_app} | set(q_entities.get("apps", []))) - {"unknown"}
        for c in cands.values():
            row = self.docs.iloc[c["row"]]
            boosts = []
            if q_apps & self.apps_of(row):
                boosts.append(("same_app", config.BOOST_SAME_APP))
            if q_codes & set(row["entities"].get("error_codes", [])):
                boosts.append(("shared_error_code", config.BOOST_SHARED_ERROR))
            if category and row["category"] == category:
                boosts.append(("same_category", config.BOOST_SAME_CATEGORY))
            if pd.notna(row["resolved_dt"]) and row["resolved_dt"] >= recent_cut:
                boosts.append(("recent", config.BOOST_RECENT))
            fb = (feedback_adj or {}).get(row["doc_id"], 0.0)
            if fb:
                boosts.append(("analyst_feedback", fb))
            c["boosts"] = {k: round(v, 3) for k, v in boosts}
            c["fused"] = c["rrf"] * (1.0 + sum(v for _, v in boosts))

    # ------------------------------------------------------------------ 5. rerank
    def rerank(self, query_text: str, cands: list[dict]) -> None:
        if not cands:
            return
        pairs = [(query_text, self.docs.iloc[c["row"]]["text"]) for c in cands]
        with self._ce_lock:
            logits = np.asarray(self.reranker.predict(pairs, batch_size=32, show_progress_bar=False), dtype=float)
        for c, lg in zip(cands, logits):
            c["rerank_logit"] = float(lg)
            c["score"] = float(_sigmoid(config.RERANK_PLATT_A * lg + config.RERANK_PLATT_B))

    @staticmethod
    def apps_of(row) -> set[str]:
        """A ticket has one application; a KB article applies to several (applies_to)."""
        return {a.strip() for a in str(row["applies_to"] or row["application"]).split(",") if a.strip()}

    # ------------------------------------------------------------------ 7. explain
    def explain(self, row, q_entities: dict, q_app: str) -> list[str]:
        why = []
        e = row["entities"]
        shared_apps = ({q_app} | set(q_entities.get("apps", []))) & self.apps_of(row)
        if shared_apps:
            why.append(f"same app ({', '.join(sorted(shared_apps))})")
        for code in sorted(set(q_entities.get("error_codes", [])) & set(e.get("error_codes", []))):
            why.append(f"same error {code}")
        for trig in sorted(set(q_entities.get("triggers", [])) & set(e.get("triggers", []))):
            why.append(f"both {trig}")
        shared = sorted(set(q_entities.get("symptoms", [])) & set(e.get("symptoms", [])))
        if shared:
            why.append("shared symptoms: " + ", ".join(shared))
        return why

    # ------------------------------------------------------------------ full search
    def search(self, query: dict, q_emb: np.ndarray, role: str, filters: dict | None = None,
               doc_type: str = "ticket", category: str | None = None, feedback_adj: dict | None = None,
               hidden: set | None = None, top_k: int | None = None, mode: str = "hybrid_rerank",
               cands: dict | None = None, keep_weak: bool = True) -> list[dict]:
        """Rank documents for a preprocessed query. Returns top_k result dicts (already role-filtered)."""
        top_k = top_k or (config.RETURN_TICKETS if doc_type == "ticket" else config.RETURN_KB)
        if cands is None:
            mask = self.filter_mask(role, filters, doc_type)
            cands = self.recall(query["search_text"], q_emb, mask, mode="hybrid" if mode == "hybrid_rerank" else mode)
        q_app = query["metadata"].get("application_inferred", "unknown")
        self.boost(cands, query["entities"], q_app, category, feedback_adj)
        hidden = hidden or set()
        pool = [c for c in cands.values() if self.doc_ids[c["row"]] not in hidden]
        pool.sort(key=lambda c: -c["fused"])
        if mode == "hybrid_rerank":
            pool = pool[:config.RERANK_TOP_N if doc_type == "ticket" else config.RERANK_TOP_N_KB]
            self.rerank(f"{query['title']}. {query['description']}", pool)
            pool.sort(key=lambda c: -c["score"])
        results = []
        for c in pool:
            row = self.docs.iloc[c["row"]]
            # "likely duplicate" only makes sense between tickets; KB chunks are similar / weak
            same_app = row["application"] == q_app and row["doc_type"] == "ticket"
            score = c.get("score")
            label = match_label(score, same_app) if score is not None else "unscored"
            if label == "dropped" or (label == "weak" and not keep_weak):
                continue
            doc = access.redact_doc(self.docs_by_id[row["doc_id"]], role)
            if doc is None:  # defence in depth: the pre-filter already removed it
                continue
            results.append({
                "doc_id": row["doc_id"], "doc_type": row["doc_type"], "source_id": row["source_id"],
                "section": row["section"] or None, "title": doc["title"], "application": row["application"],
                "category": row["category"], "subcategory": row["subcategory"], "status": row["status"],
                "assignment_group": row["assignment_group"], "environment": row["environment"],
                "created_at": row["created_at"], "resolved_at": row["resolved_at"] or None,
                "has_resolution": bool(doc["has_resolution"]),
                "score": round(score, 4) if score is not None else None, "label": label,
                "rerank_logit": round(c["rerank_logit"], 3) if "rerank_logit" in c else None,
                "bm25_rank": c["bm25_rank"], "dense_rank": c["dense_rank"], "rrf": round(c["rrf"], 5),
                "fused": round(c["fused"], 5), "boosts": c["boosts"],
                "why": self.explain(row, query["entities"], q_app)
                       + ([f"cross-encoder {score:.2f}"] if score is not None else []),
            })
            if len(results) >= top_k:
                break
        return results

    def nearest_labelled(self, cands: dict, n: int = 3) -> list[dict]:
        """Top-n labelled historical tickets from a recall pool (few-shot examples for LLM assist)."""
        rows = sorted(cands.values(), key=lambda c: -c["rrf"])
        out = []
        for c in rows:
            r = self.docs.iloc[c["row"]]
            if r["doc_type"] == "ticket":
                out.append({"title": r["title"], "category": r["category"], "subcategory": r["subcategory"]})
            if len(out) >= n:
                break
        return out

    def get_doc(self, doc_id: str, role: str) -> dict | None:
        doc = self.docs_by_id.get(doc_id)
        return access.redact_doc(doc, role) if doc else None

    def filter_options(self) -> dict:
        t = self.docs[self.docs["doc_type"] == "ticket"]
        return {c: sorted(t[c].dropna().unique().tolist())
                for c in ("application", "category", "status", "assignment_group", "environment")}


def calibrate(n_queries: int = 150, seed: int = 0) -> dict:
    """Fit Platt scaling of the cross-encoder logit on labelled historical pairs.

    Historical tickets are used as leave-one-out queries (their own document is hidden); a
    candidate is a positive when it shares the query's scenario_id (duplicate ground truth).
    Writes RERANK_PLATT_A / RERANK_PLATT_B to models/tuned_thresholds.json.
    """
    from sklearn.linear_model import LogisticRegression

    from ai.embeddings import embed_queries

    r = Retriever()
    scen = dataset.historical_scenarios()  # evaluation labels: used only to fit the calibration
    tickets = r.docs[r.docs["doc_type"] == "ticket"].sample(n_queries, random_state=seed)
    q_emb = embed_queries(tickets["search_text"].tolist())
    mask_all = r.filter_mask("Admin", None, "ticket")
    xs, ys = [], []
    for (_, q), e in zip(tickets.iterrows(), q_emb):
        mask = mask_all.copy()
        mask[r.row_of[q["doc_id"]]] = False
        cands = r.recall(q["search_text"], e, mask)
        r.boost(cands, q["entities"], q["application"], None)
        pool = sorted(cands.values(), key=lambda c: -c["fused"])[:config.RERANK_TOP_N]
        r.rerank(f"{q['title']}. {q['description']}", pool)
        for c in pool:
            xs.append(c["rerank_logit"])
            ys.append(int(scen.get(r.doc_ids[c["row"]]) == scen.get(q["doc_id"])))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.array(xs).reshape(-1, 1), np.array(ys))
    params = {"RERANK_PLATT_A": round(float(lr.coef_[0][0]), 5), "RERANK_PLATT_B": round(float(lr.intercept_[0]), 5)}
    config.save_tuned(params)
    config.RERANK_PLATT_A, config.RERANK_PLATT_B = params["RERANK_PLATT_A"], params["RERANK_PLATT_B"]
    print(f"reranker calibration on {len(xs)} pairs ({sum(ys)} positive): {params}")
    return params


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        print(__doc__)
