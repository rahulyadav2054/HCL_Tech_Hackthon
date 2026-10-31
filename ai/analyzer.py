"""AI orchestration for one ticket (no database, no HTTP): Analyzer.analyze(ticket, role, ...).

  preprocess -> [classify (model) || recall tickets || recall KB] -> LLM assist (close calls only)
  -> boost + rerank -> evidence pack -> generate (LLM) -> validate citations -> confidence
  -> fallback (Low / Escalate) -> (response, audit_record)

The backend (backend/service.py) resolves the user's role, supplies analyst-feedback signals
(hidden matches, ranking adjustments), writes the audit record and returns the response.

Quick check: python -m ai.analyzer "VPN not connecting after password change" "Error 809 on GlobalProtect"
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ai import config, confidence, resolve, validate
from ai.classify import get_classifier
from ai.embeddings import embed_docs, embed_queries
from ai.preprocess import TicketIn, preprocess
from ai.retrieve import Retriever

DISPLAY = {"High": "full", "Medium": "verify_first", "Low": "suppressed", "Escalate": "reference_only"}


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _timed(fn, *args):
    t0 = time.perf_counter()
    return fn(*args), _ms(t0)


def input_hash(title_masked: str, description_masked: str, application: str, environment: str) -> str:
    """Identifies 'the same ticket' across re-analyses (used to hide matches marked incorrect)."""
    return hashlib.sha256(json.dumps([title_masked, description_masked, application, environment]).encode()).hexdigest()


class Analyzer:
    def __init__(self):
        self.retriever = Retriever()
        self.classifier = get_classifier()
        self.retriever.reranker  # warm the cross-encoder, embedder and Presidio/spaCy
        embed_docs(["warm up"])
        preprocess(TicketIn(title="warm up", description="This is Jane Doe from IT, call +1 555 123 4567."))
        self._pool = ThreadPoolExecutor(max_workers=3)

    # ------------------------------------------------------------------ info
    def versions(self, generation: dict | None = None) -> dict:
        m = self.retriever.manifest
        return {
            "classifier": self.classifier.manifest.get("classifier_version"),
            "classifier_variant": self.classifier.variant,
            "index": m.get("index_version"), "index_built_at": m.get("built_at"),
            "embedding_model": config.EMBED_MODEL, "reranker": config.RERANK_MODEL,
            "reranker_calibration": {"A": config.RERANK_PLATT_A, "B": config.RERANK_PLATT_B},
            "llm_provider": (generation or {}).get("provider") or config.llm_settings()["provider"],
            "llm_model": (generation or {}).get("model") or config.llm_settings()["model"],
            "prompt_version": config.PROMPT_VERSION, "pii_engine": config.PII_ENGINE,
        }

    def info(self) -> dict:
        return {"versions": self.versions(), "index": self.retriever.manifest,
                "classifier": {k: self.classifier.manifest.get(k) for k in ("trained_at", "variant", "validation", "gate")},
                "thresholds": {"cls_conf_min": config.CLS_CONF_MIN, "cls_margin_min": config.CLS_MARGIN_MIN,
                               "match_duplicate": config.MATCH_DUPLICATE, "match_similar": config.MATCH_SIMILAR,
                               "match_weak": config.MATCH_WEAK, "band_high": config.BAND_HIGH,
                               "band_medium": config.BAND_MEDIUM},
                "catalog": self.retriever.filter_options()}

    # ------------------------------------------------------------------ search (FR-07)
    def search(self, query: str, role: str, filters: dict | None = None, top_k: int = 10,
               include_kb: bool = True) -> dict:
        t0 = time.perf_counter()
        pre = preprocess(TicketIn(title=query, description=query))
        q_emb = embed_queries([pre["search_text"]])[0]
        tickets = self.retriever.search(pre, q_emb, role, filters, "ticket", top_k=top_k)
        kb = self.retriever.search(pre, q_emb, role, filters, "kb", top_k=config.RETURN_KB) if include_kb else []
        return {"query_masked": pre["title"], "filters": filters or {}, "role": role, "results": tickets,
                "kb_results": kb, "latency_ms": _ms(t0)}

    # ------------------------------------------------------------------ analyze
    def analyze(self, ticket: dict, role: str, user_id: str = "unknown", use_llm: bool = True,
                hidden_for=None, feedback_adj: dict | None = None) -> tuple[dict, dict]:
        """Return (response, audit_record).

        hidden_for: optional callable(input_hash) -> set of doc_ids the analyst marked incorrect for this ticket.
        feedback_adj: optional {doc_id: ranking adjustment in [-0.05, 0.05]} from analyst feedback.
        """
        t_total = time.perf_counter()
        lat = {}
        analysis_id = str(uuid.uuid4())
        filters = ticket.get("filters") or {}

        t0 = time.perf_counter()
        tin = TicketIn(title=ticket["title"], description=ticket["description"],
                       application=ticket.get("application") or "unknown",
                       environment=ticket.get("environment") or "unknown")
        pre = preprocess(tin, vault_key=analysis_id)  # reversible PII map stays in memory only
        ihash = input_hash(pre["title"], pre["description"], tin.application, tin.environment)
        lat["preprocess"] = _ms(t0)

        t0 = time.perf_counter()
        emb_doc = embed_docs([pre["search_text"]])[0]
        emb_q = embed_queries([pre["search_text"]])[0]
        lat["embed"] = _ms(t0)

        # classification and retrieval recall run in parallel (security pre-filter applied first)
        r = self.retriever
        mask_t, mask_kb = r.filter_mask(role, filters, "ticket"), r.filter_mask(role, filters, "kb")
        f_cls = self._pool.submit(_timed, self.classifier.predict_model, pre, emb_doc)
        f_t = self._pool.submit(_timed, r.recall, pre["search_text"], emb_q, mask_t)
        f_kb = self._pool.submit(_timed, r.recall, pre["search_text"], emb_q, mask_kb)
        (cls_m, ms_cls), (cand_t, ms_t), (cand_kb, ms_kb) = f_cls.result(), f_t.result(), f_kb.result()

        t0 = time.perf_counter()  # LLM second opinion only on close calls
        classification = self.classifier.finalize(cls_m, pre, r.nearest_labelled(cand_t), use_llm=use_llm)
        lat["classify"] = ms_cls + _ms(t0)
        cat = classification["category"]["label"]

        t0 = time.perf_counter()  # boosts (incl. predicted category + feedback) and cross-encoder rerank
        hidden = set(hidden_for(ihash)) if hidden_for else set()
        similar = r.search(pre, emb_q, role, filters, "ticket", category=cat, feedback_adj=feedback_adj,
                           hidden=hidden, cands=cand_t)
        kb = r.search(pre, emb_q, role, filters, "kb", category=cat, feedback_adj=feedback_adj, hidden=hidden,
                      cands=cand_kb)
        lat["retrieve"] = max(ms_t, ms_kb) + _ms(t0)

        pack = resolve.build_evidence_pack(similar, kb, r, role)
        t0 = time.perf_counter()
        generation = resolve.generate(pre, pack, use_llm=use_llm)
        lat["llm"] = _ms(t0)

        t0 = time.perf_counter()
        validation = validate.validate(generation["output"], pack)
        conf = confidence.score(classification, pack, validation, generation, pre["entities"])
        fb = confidence.fallback(conf, classification, similar, validation, pre["entities"]) \
            if conf["band"] in ("Low", "Escalate") else None
        lat["validate_and_score"] = _ms(t0)

        display = DISPLAY[conf["band"]]
        rec_full = {"paths": validation["paths"], "conflicts": validation["conflicts"],
                    "missing_information": validation["missing_information"], "display": display,
                    "generation_status": generation["status"]}
        if not pack:
            rec_full["message"] = "No supported resolution found"
        rec_shown = dict(rec_full)
        if display == "suppressed":
            rec_shown["paths"] = []
            rec_shown["suppressed_paths"] = len(validation["paths"])
            rec_shown["message"] = "Low confidence: no recommended fix. Route to the suggested group."
        elif display == "verify_first":
            rec_shown["message"] = "Verify the cited evidence before applying."
        elif display == "reference_only":
            rec_shown["message"] = "Escalate to on-call; paths shown for reference only."

        versions = self.versions(generation)
        lat["total"] = _ms(t_total)
        input_masked = {"title": pre["title"], "description": pre["description"], "application": tin.application,
                        "environment": tin.environment, "filters": filters, "entities": pre["entities"],
                        "pii_items_masked": pre["pii_masked"]}
        gen_info = {k: generation[k] for k in ("status", "provider", "model", "attempts", "errors", "cache_hit")}
        response = {
            "analysis_id": analysis_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "user": {"user_id": user_id, "role": role},
            "input": input_masked,
            "classification": classification,
            "similar_tickets": similar,
            "kb_matches": kb,
            "evidence_pack": [{k: i[k] for k in ("source_id", "kind", "section", "score", "label", "date")} for i in pack],
            "recommendation": rec_shown,
            "validation": {k: validation[k] for k in ("total_citations", "valid_citations", "citation_pass_rate",
                                                      "removed_citations", "dropped_paths", "hallucinated_ids")},
            "confidence": conf,
            "fallback": fb,
            "generation": gen_info,
            "versions": versions,
            "latency_ms": lat,
        }
        audit_record = {
            "analysis_id": analysis_id, "created_at": datetime.now(), "user_id": user_id, "user_role": role,
            "input_hash": ihash, "input_masked": input_masked, "classification_json": classification,
            "retrieved_json": similar + kb, "evidence_ids": [i["doc_id"] for i in pack],
            "recommendation_json": rec_full, "validation_json": {**validation, "generation": gen_info},
            "confidence_json": conf, "fallback_json": fb, "band": conf["band"], "versions": versions,
            "latency_ms": lat,
        }
        return response, audit_record


_analyzer = None


def get_analyzer() -> Analyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer()
    return _analyzer


if __name__ == "__main__":
    title = sys.argv[1] if len(sys.argv) > 1 else "VPN not connecting after password change"
    desc = sys.argv[2] if len(sys.argv) > 2 else "Error 809 on GlobalProtect from home."
    out, _ = get_analyzer().analyze({"title": title, "description": desc, "environment": "Production"}, "L1")
    print(json.dumps(out, indent=2, default=str))
