"""Confidence and fallback evaluation (section 15, FR-05).

Test data (from test_labels.test_type): 25 novel + 15 vague + 10 contradictory + 50 standard tickets.
Each ticket has an expected_behavior in the dataset:
  novel          low_confidence_escalate                  -> success = band Low or Escalate
  vague          low_confidence_ask_clarifying_questions  -> success = band Low/Escalate with questions
  contradictory  medium_or_low_flag_conflicting_evidence  -> success = band Medium/Low/Escalate or conflict flagged
  standard       confident_answer_expected                -> false escalation = Low, or Escalate without P1/security
Also: routing accuracy of the fallback vs expected_assignment_group, usefulness proxy by band (path 1
cites a qrels-relevant doc), a reliability curve, and a sweep of score weights / Medium threshold over the
recorded signals (reported, not auto-applied).

Usage: python -m ai.eval.eval_confidence
"""
from __future__ import annotations

import itertools
from collections import Counter

from ai.eval import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai import config, dataset
from backend.service import get_service

GROUPS = {"novel_25": "novel", "vague_15": "vague", "contradictory_10": "contradictory", "normal_50": "standard"}


def success(row) -> bool:
    if row["group"] in ("novel", "vague"):
        return row["band"] in ("Low", "Escalate")
    if row["group"] == "contradictory":
        return row["band"] in ("Medium", "Low", "Escalate") or row["conflict_flagged"]
    return row["band"] in ("High", "Medium") or (row["band"] == "Escalate" and row["label_priority"] == "P1")


def main():
    tickets, labels, qrels, sets = common.test_tickets(), dataset.test_labels(), dataset.qrels(), dataset.test_sets()
    svc = get_service()
    rows = []
    for key, group in GROUPS.items():
        for tid in sets[key]:
            res = svc.analyze(common.as_input(tickets.loc[tid]), "analyst1")
            paths = common.generated_paths(res)
            c, fb = res["confidence"], res["fallback"] or {}
            conflicts = str(res["recommendation"].get("conflicts") or "none").strip().lower()
            rows.append({"ticket_id": tid, "group": group, "band": c["band"], "score": c["score"], **c["signals"],
                         "rules": ",".join(c["rules_fired"]), "reasons": "; ".join(c["reasons"]),
                         "label_priority": labels.loc[tid, "priority"],
                         "pred_priority": res["classification"]["priority"]["label"],
                         "uncertain": res["classification"]["uncertain"], "has_paths": bool(paths),
                         "conflict_flagged": conflicts not in ("", "none", "no", "n/a") or "conflicting_evidence" in c["rules_fired"],
                         "path1_correct": common.path1_correct(paths, qrels.get(tid, set())),
                         "escalation_group": fb.get("escalation_group"),
                         "expected_group": labels.loc[tid, "expected_assignment_group"],
                         "has_questions": bool(fb.get("clarifying_questions")),
                         "generator": res["generation"]["model"],
                         "hard_low": any(r in c["rules_fired"] for r in ("low_retrieval", "low_validity",
                                                                          "conflicting_evidence", "invalid_llm_output")),
                         "title": res["input"]["title"]})
    df = pd.DataFrame(rows)
    df["success"] = df.apply(success, axis=1)
    df["good"] = (df["group"] == "standard") & (df["path1_correct"] == True)  # noqa: E712
    flagged = df["band"].isin(["Low", "Escalate"])
    bad = df["group"].isin(["novel", "vague"])
    std = df[df["group"] == "standard"]
    expected_escalate = std["label_priority"] == "P1"
    routed = df[df["escalation_group"].notna()]
    res = {
        "n": len(df), "generators": dict(Counter(df["generator"].fillna("not called"))),
        "expected_behavior_success": {g: round(float(df[df["group"] == g]["success"].mean()), 4) for g in GROUPS.values()},
        "escalation_recall_novel_vague": round(float(flagged[bad].mean()), 4),
        "escalation_recall_by_group": {g: round(float(flagged[df["group"] == g].mean()), 4) for g in GROUPS.values()},
        "vague_with_clarifying_questions": round(float(df[df["group"] == "vague"]["has_questions"].mean()), 4),
        "contradictory_conflict_flagged": round(float(df[df["group"] == "contradictory"]["conflict_flagged"].mean()), 4),
        "bands_by_group": {g: df[df["group"] == g]["band"].value_counts().to_dict() for g in GROUPS.values()},
        "standard": {
            "false_low_rate": round(float((std["band"] == "Low").mean()), 4),
            "false_escalate_rate": round(float(((std["band"] == "Escalate") & ~expected_escalate).mean()), 4),
            "false_escalation_rate_total": round(float(((std["band"] == "Low") |
                                                        ((std["band"] == "Escalate") & ~expected_escalate)).mean()), 4),
            "p1_tickets": int(expected_escalate.sum()),
            "p1_escalated": int(((std["band"] == "Escalate") & expected_escalate).sum())},
        "routing_accuracy": {"n": int(len(routed)),
                             "matches_expected_group": round(float((routed["escalation_group"] == routed["expected_group"]).mean()), 4)
                             if len(routed) else None},
        "usefulness_proxy_by_band": {b: {"n_with_paths": int(g["path1_correct"].notna().sum()),
                                         "path1_relevant_rate": round(float(g["path1_correct"].dropna().astype(float).mean()), 4)
                                         if g["path1_correct"].notna().any() else None}
                                     for b, g in df.groupby("band")},
    }

    bins = [0, 0.4, 0.55, 0.65, 0.75, 0.85, 1.01]
    df["bin"] = pd.cut(df["score"], bins, right=False)
    rel = df.groupby("bin", observed=False).agg(n=("good", "size"), good_rate=("good", "mean"),
                                                mean_score=("score", "mean")).reset_index()
    res["reliability"] = [{"bin": str(b), "n": int(n), "good_rate": None if pd.isna(g) else round(float(g), 4),
                           "mean_score": None if pd.isna(m) else round(float(m), 4)}
                          for b, n, g, m in rel[["bin", "n", "good_rate", "mean_score"]].itertuples(index=False)]

    y = df["good"].to_numpy()
    sig = df[["C", "R", "A", "V"]].to_numpy()
    hard = df["hard_low"].to_numpy() | (df["band"] == "Escalate").to_numpy()
    sweep = []
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    for wc, wr, wa in itertools.product(grid, grid, grid):
        wv = round(1 - wc - wr - wa, 2)
        if wv < 0.05:
            continue
        s = sig @ np.array([wc, wr, wa, wv])
        for thr in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            ok = (s >= thr) & ~hard
            tp, fp, fn = int((ok & y).sum()), int((ok & ~y).sum()), int((~ok & y).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            sweep.append({"wC": float(wc), "wR": float(wr), "wA": float(wa), "wV": wv, "medium_threshold": thr,
                          "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
    sweep.sort(key=lambda r: (-r["f1"], -r["precision"]))
    target = (config.W_CLASSIFICATION, config.W_RETRIEVAL, config.W_AGREEMENT, config.W_VALIDITY, config.BAND_MEDIUM)
    default = next((r for r in sweep if all(abs(a - b) < 1e-6 for a, b in zip(
        (r["wC"], r["wR"], r["wA"], r["wV"], r["medium_threshold"]), target))), None)
    res["weight_sweep"] = {"default": default, "top5": sweep[:5], "configs_tried": len(sweep),
                           "note": "proxy target (standard ticket with a qrels-relevant path 1); reported, not auto-applied"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"High": "#2e7d32", "Medium": "#f9a825", "Low": "#c62828", "Escalate": "#6a1b9a"}
    bottom = np.zeros(len(GROUPS))
    for b in colors:
        vals = np.array([res["bands_by_group"][g].get(b, 0) for g in GROUPS.values()])
        axes[0].bar(list(GROUPS.values()), vals, bottom=bottom, label=b, color=colors[b])
        bottom += vals
    axes[0].set_title("Confidence band by test group")
    axes[0].legend(fontsize=8)
    pts = [r for r in res["reliability"] if r["n"]]
    axes[1].plot([r["mean_score"] for r in pts], [r["good_rate"] for r in pts], "o-", label="observed")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="ideal")
    for r in pts:
        axes[1].annotate(f"n={r['n']}", (r["mean_score"], r["good_rate"]), fontsize=7, xytext=(3, 3),
                         textcoords="offset points")
    axes[1].set_xlabel("confidence score")
    axes[1].set_ylabel("share of good answers")
    axes[1].set_title("Reliability curve")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(common.REPORTS / "confidence_bands.png", dpi=130)
    plt.close(fig)

    res["errors"] = df[~df["success"]][["ticket_id", "group", "band", "score", "R", "A", "V", "title", "reasons"]] \
        .head(10).to_dict("records")
    res["per_ticket"] = df.drop(columns=["bin"]).to_dict("records")
    common.save_json("confidence_results.json", res)
    print({k: v for k, v in res.items() if k not in ("per_ticket", "weight_sweep", "errors", "reliability")})
    return res


if __name__ == "__main__":
    main()
