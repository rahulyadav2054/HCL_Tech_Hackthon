"""Latency evaluation (section 15): p50 / p95 per stage.

The LLM response cache is disabled so API calls are really timed. With an API key configured the
script compares the API LLM with the local offline generator ("local vs API LLM").

Usage: python -m ai.eval.eval_latency [--runs 50] [--api-runs 20]
"""
from __future__ import annotations

import argparse
import os

os.environ["LLM_CACHE"] = "0"

from ai.eval import common  # noqa: E402
import numpy as np  # noqa: E402

from ai import config, dataset, llm  # noqa: E402
from backend.service import get_service  # noqa: E402


def run(svc, ids, tickets, use_llm: bool) -> dict:
    lat, gens = [], []
    for tid in ids:
        res = svc.analyze(common.as_input(tickets.loc[tid]), "analyst1", use_llm=use_llm)
        lat.append(res["latency_ms"])
        gens.append(res["generation"]["model"])
    stages = sorted({k for l in lat for k in l})
    return {"stages_ms": {s: {"p50": int(np.percentile([l.get(s, 0) for l in lat], 50)),
                              "p95": int(np.percentile([l.get(s, 0) for l in lat], 95))} for s in stages},
            "generators": {g or "not called": gens.count(g) for g in set(gens)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--api-runs", type=int, default=20, help="API calls are rate-limited/billed: fewer by default")
    args = ap.parse_args()
    tickets, sets = common.test_tickets(), dataset.test_sets()
    ids = (sets["normal_50"] * 2)[:args.runs]
    svc = get_service()
    svc.analyze(common.as_input(tickets.loc[ids[0]]), "analyst1", write_audit=False)  # warm-up, not timed
    res = {"runs": len(ids), "hardware": f"CPU only, {os.cpu_count()} logical cores"}
    res["offline_generator"] = {"model": "offline-extractive", **run(svc, ids, tickets, use_llm=False)}
    if llm.available() and args.api_runs:
        res["api_llm"] = {"model": config.LLM_MODEL, "runs": args.api_runs,
                          **run(svc, ids[:args.api_runs], tickets, use_llm=True)}
    common.save_json("latency_results.json", res)
    print(res)
    return res


if __name__ == "__main__":
    main()
