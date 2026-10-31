"""Duplicate / similar-ticket retrieval evaluation (section 15, FR-02).

Queries: the 400 `standard` test tickets. Relevance: data/labels/test_qrels.csv (historical tickets and KB
articles from the same root cause). Hard negatives: data/labels/test_hard_negatives.csv (lookalike tickets
with a different root cause that should NOT rank high).
Metrics over historical tickets: Recall@1, Recall@5 (hit rate), MRR@10, nDCG@5, P@5 and the hard-negative
rate in the top 5. KB: Recall@3 (a relevant approved article among the top 3 KB articles).
Ablation: BM25 only vs dense only vs hybrid (RRF + boosts) vs hybrid + cross-encoder rerank.

--tune: re-check the duplicate threshold on held-in historical tickets (leave-one-out, never the test set).

Usage: python -m ai.eval.eval_retrieval [--tune] [--limit N]
"""
from __future__ import annotations

import argparse
import math

from ai.eval import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai import config, dataset
from ai.embeddings import embed_queries
from ai.preprocess import load_processed
from ai.retrieve import Retriever

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def ranked(r: Retriever, q: dict, emb, mode: str, mask, k: int = 10, doc_type: str = "ticket") -> list[dict]:
    """Full ranking for one mode (no label/threshold dropping, so ranking metrics are comparable)."""
    cands = r.recall(q["search_text"], emb, mask, mode="hybrid" if mode == "hybrid_rerank" else mode)
    if mode == "bm25":
        pool = sorted(cands.values(), key=lambda c: c["bm25_rank"])
    elif mode == "dense":
        pool = sorted(cands.values(), key=lambda c: c["dense_rank"])
    else:
        r.boost(cands, q["entities"], q["application"], None)
        pool = sorted(cands.values(), key=lambda c: -c["fused"])
        if mode == "hybrid_rerank":
            pool = pool[:config.RERANK_TOP_N if doc_type == "ticket" else config.RERANK_TOP_N_KB]
            r.rerank(f"{q['title']}. {q['description']}", pool)
            pool.sort(key=lambda c: -c["score"])
    return [{"doc_id": r.doc_ids[c["row"]], "score": c.get("score"), "row": c["row"]} for c in pool[:k]]


def metrics_for(results: list[str], relevant: set[str], hard: set[str]) -> dict:
    rel = [d in relevant for d in results]
    first = next((i for i, x in enumerate(rel) if x), None)
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(rel[:5]) if x)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
    top5 = results[:5]
    return {"r1": float(bool(rel[:1] and rel[0])), "r5": float(any(rel[:5])),
            "mrr": 1 / (first + 1) if first is not None else 0.0, "ndcg5": dcg / idcg if idcg else 0.0,
            "p5": sum(rel[:5]) / 5, "hard_neg": sum(d in hard for d in top5) / max(1, len(top5)),
            "has_hard": bool(hard)}


def tune_duplicate_threshold(r: Retriever, n: int = 150) -> dict:
    scen = dataset.historical_scenarios()
    hist = r.docs[r.docs["doc_type"] == "ticket"].sample(n, random_state=11)
    embs = embed_queries(hist["search_text"].tolist())
    pairs = []
    mask_all = r.filter_mask(common.ADMIN_ROLE, None, "ticket")
    for (_, q), e in zip(hist.iterrows(), embs):
        mask = mask_all.copy()
        mask[r.row_of[q["doc_id"]]] = False
        qd = {"search_text": q["search_text"], "entities": q["entities"], "application": q["application"],
              "title": q["title"], "description": q["description"]}
        for c in ranked(r, qd, e, "hybrid_rerank", mask, k=config.RERANK_TOP_N):
            same_app = r.docs.iloc[c["row"]]["application"] == q["application"]
            pairs.append((c["score"], same_app, scen.get(c["doc_id"]) == scen.get(q["doc_id"])))
    sweep = []
    for t in np.arange(0.50, 0.991, 0.01):
        sel = [rel for s, a, rel in pairs if s >= t and a]
        if len(sel) >= 10:
            sweep.append({"threshold": round(float(t), 2), "precision": round(float(np.mean(sel)), 4), "n": len(sel)})
    # The match score is Platt-calibrated, so one pair's score is itself P(same issue). We therefore only
    # RAISE the configured threshold when its precision misses the 0.9 target, never lower it: a lower
    # threshold would label pairs that are individually ~60% likely as "likely duplicate".
    at_cfg = next((s for s in sweep if s["threshold"] >= config.MATCH_DUPLICATE - 1e-9), None)
    if at_cfg is None or at_cfg["precision"] >= 0.9:
        chosen = config.MATCH_DUPLICATE
    else:
        ok = [s for s in sweep if s["threshold"] > config.MATCH_DUPLICATE and s["precision"] >= 0.9]
        chosen = ok[0]["threshold"] if ok else config.MATCH_DUPLICATE
    sim = [rel for s, a, rel in pairs if config.MATCH_SIMILAR <= s]
    weak = [rel for s, a, rel in pairs if config.MATCH_WEAK <= s < config.MATCH_SIMILAR]
    return {"queries": n, "pairs": len(pairs), "duplicate_threshold": chosen, "configured": config.MATCH_DUPLICATE,
            "lowest_threshold_meeting_target": next((s["threshold"] for s in sweep if s["precision"] >= 0.9), None),
            "precision_at_configured": at_cfg["precision"] if at_cfg else None,
            "precision_similar_or_better": round(float(np.mean(sim)), 4) if sim else None,
            "precision_weak_band": round(float(np.mean(weak)), 4) if weak else None, "sweep": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    r = Retriever()
    proc = load_processed("test").set_index("ticket_id", drop=False)
    qrels, hard = dataset.qrels(), dataset.hard_negatives()
    ids = [t for t in dataset.test_sets()["standard"] if t in qrels]
    if args.limit:
        ids = ids[:args.limit]
    mask_t = r.filter_mask(common.ADMIN_ROLE, None, "ticket")
    mask_kb = r.filter_mask(common.ADMIN_ROLE, None, "kb")
    embs = embed_queries(proc.loc[ids, "search_text"].tolist())

    per_mode = {m: [] for m in MODES}
    kb_hits, worst = [], []
    for tid, e in zip(ids, embs):
        q = proc.loc[tid]
        qd = {"search_text": q["search_text"], "entities": q["entities"], "application": q["application"],
              "title": q["title"], "description": q["description"]}
        rel_tickets = {d for d in qrels[tid] if not d.startswith("KB")}
        rel_kb = {d for d in qrels[tid] if d.startswith("KB")}
        for m in MODES:
            res = ranked(r, qd, e, m, mask_t)
            met = metrics_for([x["doc_id"] for x in res], rel_tickets, hard.get(tid, set()))
            per_mode[m].append(met)
            if m == "hybrid_rerank":
                worst.append({"ticket_id": tid, "title": q["title"], "mrr": met["mrr"],
                              "top1": res[0]["doc_id"] if res else None,
                              "top1_relevant": bool(res and res[0]["doc_id"] in rel_tickets),
                              "top1_hard_negative": bool(res and res[0]["doc_id"] in hard.get(tid, set()))})
        if rel_kb:
            kb_res = ranked(r, qd, e, "hybrid_rerank", mask_kb, k=10, doc_type="kb")
            top_kb = list(dict.fromkeys(r.docs.iloc[x["row"]]["source_id"] for x in kb_res))[:config.RETURN_KB]
            kb_hits.append(bool(rel_kb & set(top_kb)))

    def agg(rows):
        out = {k: round(float(np.mean([x[k] for x in rows])), 4) for k in ("r1", "r5", "mrr", "ndcg5", "p5", "hard_neg")}
        withhn = [x for x in rows if x["has_hard"]]
        out["hard_neg_queries_with_lookalikes"] = round(float(np.mean([x["hard_neg"] for x in withhn])), 4) if withhn else None
        out["n"] = len(rows)
        return out

    results = {"n_queries": len(ids), "role": f"{common.ADMIN_ROLE} (all sensitivities)",
               "by_mode": {m: agg(v) for m, v in per_mode.items()},
               "by_mode_queries_with_hard_negatives": {m: agg([x for x in v if x["has_hard"]]) for m, v in per_mode.items()},
               "kb_recall_at_3": round(float(np.mean(kb_hits)), 4) if kb_hits else None,
               "kb_queries": len(kb_hits),
               "worst": sorted(worst, key=lambda w: w["mrr"])[:10],
               "n_mrr_below_1": sum(1 for w in worst if w["mrr"] < 1)}
    if args.tune:
        t = tune_duplicate_threshold(r)
        results["threshold_tuning"] = t
        if t["duplicate_threshold"] != config.MATCH_DUPLICATE:
            config.save_tuned({"MATCH_DUPLICATE": t["duplicate_threshold"]})
            print(f"raised MATCH_DUPLICATE -> {t['duplicate_threshold']} (saved)")
        else:
            print(f"duplicate threshold {config.MATCH_DUPLICATE}: precision {t['precision_at_configured']} (target >= 0.9)")

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["r1", "r5", "mrr", "ndcg5", "hard_neg"]
    x = np.arange(len(keys))
    for i, m in enumerate(MODES):
        ax.bar(x + i * 0.2 - 0.3, [results["by_mode"][m][k] for k in keys], width=0.2, label=m)
    ax.set_xticks(x, ["Recall@1", "Recall@5", "MRR", "nDCG@5", "hard-neg rate@5"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title(f"Retrieval ablation (n={len(ids)} standard test tickets, qrels)")
    fig.tight_layout()
    fig.savefig(common.REPORTS / "retrieval_ablation.png", dpi=130)
    plt.close(fig)
    common.save_json("retrieval_results.json", results)
    for m in MODES:
        print(m, results["by_mode"][m])
    print("KB recall@3", results["kb_recall_at_3"])
    return results


if __name__ == "__main__":
    main()
