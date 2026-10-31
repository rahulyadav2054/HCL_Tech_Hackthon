"""Resolution usefulness evaluation (section 15, FR-03 / FR-04).

Runs the full pipeline (backend service) on the 25 rating tickets and reports:
  * citation pass rate, share of recommendations with >= 1 valid citation, hallucinated IDs blocked
  * with vs without citation validation: unsupported citations / paths an analyst would have seen
    without ai/validate.py
  * automatic usefulness proxies: path 1 cites a doc the qrels mark relevant; and token overlap between
    path 1's steps and the dataset's reference_resolution (rapidfuzz token_set_ratio, 0-100)
  * human rating: blind rating sheets (no confidence band shown) for 2 raters in data/labels/ratings/.
    Once filled (rubric 1-5), re-running reports means and quadratic-weighted Cohen's kappa.
    No ratings are invented by the script.

Usage: python -m ai.eval.eval_resolution
"""
from __future__ import annotations

from collections import Counter

from ai.eval import common
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.metrics import cohen_kappa_score

from ai import config, dataset
from backend import audit
from backend.service import get_service

RATING_DIR = config.LABELS_DIR / "ratings"
DIMS = ("relevance", "actionability", "evidence_support")


def write_rating_sheets(rows: list[dict]):
    RATING_DIR.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame(rows)
    for d in DIMS:
        sheet[d] = ""
    sheet["comment"] = ""
    for rater in ("rater1", "rater2"):
        path = RATING_DIR / f"{rater}.csv"
        if path.exists():
            existing = pd.read_csv(path, dtype=str, keep_default_na=False)
            if any((existing[d] != "").any() for d in DIMS if d in existing):
                continue  # never overwrite ratings that were already entered
        sheet.to_csv(path, index=False)
    (RATING_DIR / "README.md").write_text(
        "# Resolution rating sheets\n\nTwo raters score path 1 of each ticket independently, blind to the confidence "
        "band (the sheet does not show it).\n\nRubric (1-5):\n\n| Score | Relevance | Actionability | Evidence support |\n"
        "|---|---|---|---|\n| 5 | Fixes this exact issue | Steps can be followed as written | Every step backed by a cited source |\n"
        "| 3 | Plausible, partly fits | Needs some interpretation | Some steps uncited |\n"
        "| 1 | Wrong issue | Vague or unsafe | Citations missing or wrong |\n\nFill `relevance`, `actionability`, "
        "`evidence_support` with 1-5, then run `python -m ai.eval.eval_resolution` to compute means and agreement.\n",
        encoding="utf-8")


def read_ratings() -> dict | None:
    paths = [RATING_DIR / "rater1.csv", RATING_DIR / "rater2.csv"]
    if not all(p.exists() for p in paths):
        return None
    a, b = (pd.read_csv(p, dtype=str, keep_default_na=False) for p in paths)
    out = {}
    for d in DIMS:
        both = [(x, y) for x, y in zip(a[d], b[d]) if x.strip() and y.strip()]
        if not both:
            return None
        x = np.array([int(v) for v, _ in both])
        y = np.array([int(v) for _, v in both])
        out[d] = {"n": len(both), "mean_rater1": round(x.mean(), 2), "mean_rater2": round(y.mean(), 2),
                  "mean": round((x.mean() + y.mean()) / 2, 2),
                  "weighted_kappa": round(cohen_kappa_score(x, y, weights="quadratic"), 3) if len(set(x) | set(y)) > 1 else None}
    return out


def main():
    tickets, labels, qrels = common.test_tickets(), dataset.test_labels(), dataset.qrels()
    svc = get_service()
    rows, per = [], []
    for tid in dataset.test_sets()["rating_25"]:
        res = svc.analyze(common.as_input(tickets.loc[tid]), "analyst1")
        val = audit.get_analysis(res["analysis_id"])["validation_json"]
        paths = common.generated_paths(res)
        ref = labels.loc[tid, "reference_resolution"]
        overlap = fuzz.token_set_ratio(" ".join(paths[0]["steps"]).lower(), ref.lower()) if paths else None
        per.append({"ticket_id": tid, "band": res["confidence"]["band"], "pack": len(res["evidence_pack"]),
                    "generator": res["generation"]["model"], "total_citations": val["total_citations"],
                    "valid_citations": val["valid_citations"], "removed": len(val["removed_citations"]),
                    "hallucinated": len(val["hallucinated_ids"]), "dropped_paths": len(val["dropped_paths"]),
                    "paths": len(paths), "requires_approval": sum(bool(p.get("requires_approval")) for p in paths),
                    "path1_correct": common.path1_correct(paths, qrels.get(tid, set())),
                    "reference_overlap": overlap})
        if paths:
            p = paths[0]
            rows.append({"ticket_id": tid, "title": res["input"]["title"], "description": res["input"]["description"],
                         "path_title": p["title"], "steps": " | ".join(p["steps"]),
                         "evidence": " | ".join(f"{e['source_id']}: \"{e['excerpt']}\" ({e.get('why', '')})"
                                                for e in p["evidence"])})
    df = pd.DataFrame(per)
    total, valid = int(df["total_citations"].sum()), int(df["valid_citations"].sum())
    with_pack = df[df["pack"] > 0]
    res = {
        "n_tickets": len(df), "generators": dict(Counter(df["generator"].fillna("not called"))),
        "llm": config.llm_settings()["model"],
        "with_evidence_pack": int(len(with_pack)),
        "citation_pass_rate": round(valid / total, 4) if total else None,
        "recommendations_with_valid_citation": round(float((with_pack["valid_citations"] > 0).mean()), 4) if len(with_pack) else None,
        "hallucinated_ids_blocked": int(df["hallucinated"].sum()),
        "excerpts_removed": int(df["removed"].sum()),
        "paths_dropped_no_valid_evidence": int(df["dropped_paths"].sum()),
        "without_validation": {
            "unsupported_citations_shown": int(df["removed"].sum()),
            "unsupported_citation_rate": round(float(df["removed"].sum() / total), 4) if total else None,
            "paths_shown_without_any_valid_evidence": int(df["dropped_paths"].sum())},
        "with_validation": {"unsupported_citations_shown": 0, "paths_shown_without_any_valid_evidence": 0},
        "path1_relevant_rate": round(float(df["path1_correct"].dropna().astype(float).mean()), 4)
        if df["path1_correct"].notna().any() else None,
        "reference_overlap_mean": round(float(df["reference_overlap"].dropna().mean()), 1)
        if df["reference_overlap"].notna().any() else None,
        "paths_flagged_requires_approval": int(df["requires_approval"].sum()),
        "bands": df["band"].value_counts().to_dict(),
        "per_ticket": per,
    }
    write_rating_sheets(rows)
    ratings = read_ratings()
    res["human_rating"] = ratings or {"status": f"pending: fill data/labels/ratings/rater1.csv and rater2.csv "
                                                f"({len(rows)} rows each), then re-run"}
    common.save_json("resolution_results.json", res)
    print({k: v for k, v in res.items() if k != "per_ticket"})
    return res


if __name__ == "__main__":
    main()
