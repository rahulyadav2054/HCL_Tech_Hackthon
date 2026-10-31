"""Run every evaluation, then assemble reports/evaluation_report.md and reports/evaluation_summary.json.

Usage:
  python -m ai.eval.run_all                 run all evaluations + build the report
  python -m ai.eval.run_all --report-only   rebuild the report from the existing reports/*.json
  python -m ai.eval.run_all --llm           also run the classification LLM-assist comparison
Scripts run one at a time (each loads the models; the build laptop has ~8 GB RAM).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SCRIPTS = [("eval_classification", []), ("eval_retrieval", ["--tune"]), ("eval_resolution", []),
           ("eval_confidence", []), ("eval_traceability", []), ("eval_latency", []), ("worked_examples", [])]


def run_scripts(llm: bool):
    (ROOT / "data" / "eval_audit.sqlite").unlink(missing_ok=True)  # fresh evaluation audit log
    for script, args in SCRIPTS:
        if script == "eval_classification" and llm:
            args = ["--llm"]
        print(f"\n=== {script} {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, "-m", f"ai.eval.{script}", *args], cwd=ROOT, check=True)


def load(name):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def f3(v):
    return "n/a" if v is None else f"{v:.3f}"


def md(rows, cols=None):
    if not rows:
        return "_none_\n"
    cols = cols or list(rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return "" if v is None else str(v).replace("|", "/").replace("\n", " ")
    return ("| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
            + "".join("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |\n" for r in rows))


def tag_classification(e):
    if e["kind"] == "priority":
        return "priority: no scope signal in text" if not e.get("signals") else "priority: ambiguous scope signal"
    return "wrong category (keyword lookalike / ambiguous ticket)"


def tag_retrieval(w):
    if w["top1_hard_negative"]:
        return "keyword lookalike (labelled hard negative ranked first)"
    if not w["top1_relevant"]:
        return "data gap / ambiguous ticket"
    return "threshold"


def tag_confidence(e):
    return {"vague": "ambiguous ticket scored above Low", "novel": "threshold (unrelated case scored >= 0.60)",
            "contradictory": "conflict not detected (one root cause dominated the evidence)"}.get(
        e["group"], "threshold (standard ticket scored Low)")


def build_report():
    sys.path.insert(0, str(ROOT))
    from ai import config, dataset
    cls, ret, resl, conf = (load(f"{n}_results.json") for n in ("classification", "retrieval", "resolution", "confidence"))
    trace, lat = load("traceability_results.json"), load("latency_results.json")
    s = dataset.summary()
    idx = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    cman = json.loads((config.MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    L = []
    w = L.append

    w("# Evaluation report - IT Ticket Resolution Assistant\n")
    w(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m ai.eval.run_all`. Every number below is read from "
      f"`reports/*.json`, produced by the scripts in `ai/eval/`. Configured LLM: **{config.llm_settings()['model']}** "
      f"({config.LLM_PROVIDER}); the generator actually used per analysis is listed with each result.\n")

    w("## 1. Dataset\n")
    w(f"* **it-ticket-dataset** (synthetic; `data/DATA_CARD.md`): {s['historical']} historical tickets (index + classifier "
      f"training, Jan 2025 - Jun 2026), {s['test']} test tickets (Jul - Oct 2026, input fields only), "
      f"{s['kb_articles']} KB articles ({s['kb_approved']} approved; drafts/superseded are never retrieved).")
    w(f"* Test types: {s['test_types']}. Ground truth: `test_labels.csv` (category, subcategory, priority, expected "
      "behaviour, expected group, reference fix), `test_qrels.csv` (relevant historical tickets/KB), "
      "`test_hard_negatives.csv` (lookalikes that must not rank high). Scenario ids are used only for evaluation and "
      "for fitting the reranker calibration on historical pairs; they are never model features.")
    w(f"* Historical status {s['historical_status']}; sensitivity {s['historical_sensitivity']}.")
    w(f"* **Index:** {idx['ticket_docs']} historical tickets ({idx['tickets_with_resolution']} with a resolution) "
      f"+ {idx['kb_chunks']} sections of {idx['kb_articles']} approved KB articles; `{idx['embedding_model']}`, "
      f"FAISS `{idx['faiss_index']}`, index version `{idx['index_version']}`.\n")

    w("## 2. Method\n")
    w("Pipeline per ticket: preprocess (PII masking with Presidio + regex, entity extraction) -> calibrated "
      "logistic-regression classifiers (category, subcategory restricted to the taxonomy, priority + rule overrides) "
      "in parallel with hybrid recall (BM25 top 50 + dense top 50, Reciprocal Rank Fusion, metadata boosts) -> "
      "cross-encoder rerank (match score = Platt-calibrated sigmoid of the logit, fitted on labelled historical pairs: "
      f"A={config.RERANK_PLATT_A}, B={config.RERANK_PLATT_B}) -> evidence pack -> grounded generation -> citation "
      "validation -> confidence score with hard rules -> fallback -> audit row.\n")
    w(f"Classifier: {cman['model']}, features `{cman['variant']}` (chosen on validation macro-F1: "
      + ", ".join(f"{k} {v['macro_f1']:.3f}" for k, v in cman["validation"].items())
      + f"). Model-only gate: conf >= {cman['gate']['conf']} and margin >= {cman['gate']['margin']} "
      f"({'workflow default kept: no setting improved validation accuracy' if cman['gate'].get('kept_default') else 'tuned on validation'}).\n")

    w("## 3. Results\n")
    if cls:
        w(f"### 3.1 Classification ({cls['n_standard']} standard test tickets)\n")
        rows = [{"target": t, "accuracy": cls[t]["accuracy"], "macro-F1": cls[t]["macro_f1"]}
                for t in ("category", "subcategory", "priority")]
        rows.append({"target": "priority (model only, no rules)", "accuracy": cls["priority_model_only"]["accuracy"],
                     "macro-F1": cls["priority_model_only"]["macro_f1"]})
        w(md(rows))
        w(f"Category + subcategory both correct: {pct(cls['subcategory_pair_accuracy'])}. **Operationally important "
          f"classes:** P1 recall {pct(cls['priority_recall']['P1'])}, P2 recall {pct(cls['priority_recall']['P2'])} "
          f"(model only: P1 {pct(cls['priority_recall_model_only']['P1'])}, P2 {pct(cls['priority_recall_model_only']['P2'])}); "
          f"support {cls['priority_support']}.\n")
        w("By test type (category):\n")
        w(md([{"test_type": k, "n": v["n"], "accuracy": v["accuracy"], "macro-F1": v["macro_f1"],
               "below model-only gate": v["low_confidence_share"]} for k, v in cls["by_test_type"].items()]))
        w("Feature ablation (category, trained on all historical tickets):\n")
        w(md([{"features": k, "accuracy": v["accuracy"], "macro-F1": v["macro_f1"]} for k, v in cls["feature_ablation_category"].items()]))
        la = cls["llm_assist"]
        if "with" in la:
            w(f"With vs without LLM assist (category): {la['without']['accuracy']:.3f} -> {la['with']['accuracy']:.3f} "
              f"accuracy; {la['calls']} close calls sent, the LLM agreed with the model's top 2 on {la['agreed']}.\n")
        else:
            w(f"LLM assist: {la['status']}; {la.get('close_calls_that_would_be_sent', 0)} standard tickets fall below the gate.\n")
        w(f"Model-only gate on standard tickets: {pct(cls['gate']['share_model_only'])} pass, accuracy "
          f"{pct(cls['gate']['accuracy_model_only'])} vs {pct(cls['gate']['accuracy_deferred'])} on deferred close calls. "
          f"Classifications with a top-term explanation: {pct(cls['explained_share'])}. Confusion matrix: "
          "`reports/classification_confusion.png`.\n")

    if ret:
        w(f"### 3.2 Duplicate retrieval ({ret['n_queries']} standard test tickets, relevance from qrels)\n")
        w(md([{"mode": m, "Recall@1": v["r1"], "Recall@5": v["r5"], "MRR": v["mrr"], "nDCG@5": v["ndcg5"],
               "P@5": v["p5"], "hard-neg rate@5": v["hard_neg"]} for m, v in ret["by_mode"].items()]))
        hq = ret["by_mode_queries_with_hard_negatives"]
        w(f"Queries that have labelled lookalikes (n = {next(iter(hq.values()))['n']}): the hard-negative rate is the share "
          "of the top 5 that are labelled hard negatives:\n")
        w(md([{"mode": m, "Recall@1": v["r1"], "MRR": v["mrr"], "nDCG@5": v["ndcg5"], "hard-neg rate@5": v["hard_neg"]}
              for m, v in hq.items()]))
        w(f"KB: a relevant approved article is among the top 3 KB articles for {pct(ret['kb_recall_at_3'])} of "
          f"{ret['kb_queries']} queries that have one.\n")
        t = ret.get("threshold_tuning")
        if t:
            w(f"Duplicate threshold check on {t['queries']} held-in historical queries ({t['pairs']} pairs, leave-one-out): "
              f"precision of 'likely duplicate' at {t['configured']} = {f3(t['precision_at_configured'])} (target >= 0.9; "
              f"lowest threshold meeting it: {t['lowest_threshold_meeting_target']}; not lowered because the calibrated "
              f"score is itself P(same issue)). Precision of pairs labelled similar-or-better {f3(t['precision_similar_or_better'])}; "
              f"weak band (greyed, never evidence) {f3(t['precision_weak_band'])}.\n")
        w("Chart: `reports/retrieval_ablation.png`.\n")

    if resl:
        w(f"### 3.3 Resolution usefulness ({resl['n_tickets']} rating tickets; generators {resl['generators']})\n")
        w(f"* With an evidence pack: {resl['with_evidence_pack']}; citation pass rate {pct(resl['citation_pass_rate'])}; "
          f"recommendations with >= 1 valid citation {pct(resl['recommendations_with_valid_citation'])}.")
        wo = resl["without_validation"]
        w(f"* **With vs without citation validation:** without `ai/validate.py` the analyst would have seen "
          f"{wo['unsupported_citations_shown']} unsupported citations ({pct(wo['unsupported_citation_rate'])}) and "
          f"{wo['paths_shown_without_any_valid_evidence']} paths with no valid evidence; with validation: 0 and 0. "
          f"Hallucinated IDs blocked: {resl['hallucinated_ids_blocked']}.")
        w(f"* Usefulness proxies: path 1 cites a qrels-relevant ticket/KB in {pct(resl['path1_relevant_rate'])} of cases; "
          f"mean token overlap with the dataset's reference fix {resl['reference_overlap_mean']}/100. "
          f"Paths flagged *requires approval*: {resl['paths_flagged_requires_approval']}.")
        hr = resl["human_rating"]
        if "status" in hr:
            w(f"* **Human rating (rubric 1-5, 2 raters, blind to band): {hr['status']}.** No ratings are invented.\n")
        else:
            w(md([{"dimension": d, "n": v["n"], "mean": v["mean"], "rater1": v["mean_rater1"], "rater2": v["mean_rater2"],
                   "weighted kappa": v["weighted_kappa"]} for d, v in hr.items()]))

    if conf:
        w(f"### 3.4 Confidence and fallback (25 novel + 15 vague + 10 contradictory + 50 standard; generators {conf['generators']})\n")
        w("Share of tickets that got their **expected behaviour** from `test_labels.csv`:\n")
        w(md([{"group": g, "success": v} for g, v in conf["expected_behavior_success"].items()]))
        st = conf["standard"]
        w(f"* Escalation recall on novel + vague: {pct(conf['escalation_recall_novel_vague'])}; vague tickets with "
          f"clarifying questions {pct(conf['vague_with_clarifying_questions'])}; contradictory tickets with a conflict "
          f"flagged {pct(conf['contradictory_conflict_flagged'])}.")
        w(f"* **False-escalation rate on standard tickets:** {pct(st['false_escalation_rate_total'])} (Low "
          f"{pct(st['false_low_rate'])}, Escalate without a P1 label {pct(st['false_escalate_rate'])}); "
          f"{st['p1_escalated']}/{st['p1_tickets']} P1 tickets escalated.")
        ra = conf["routing_accuracy"]
        w(f"* Fallback routing matches the dataset's expected_assignment_group in {pct(ra['matches_expected_group'])} "
          f"of {ra['n']} routed cases.\n")
        w(md([{"group": g, **{b: d.get(b, 0) for b in ("High", "Medium", "Low", "Escalate")}} for g, d in conf["bands_by_group"].items()]))
        w("Usefulness proxy by band (path 1 cites a qrels-relevant source):\n")
        w(md([{"band": b, "with paths": v["n_with_paths"], "path-1 relevant": v["path1_relevant_rate"]}
              for b, v in conf["usefulness_proxy_by_band"].items()]))
        w("Reliability (share of good answers per score bin):\n")
        w(md([r for r in conf["reliability"] if r["n"]]))
        ws = conf["weight_sweep"]
        top = ws["top5"][0]
        w(f"Weight / threshold sweep ({ws['configs_tried']} configurations): default 0.25/0.35/0.20/0.20 with Medium >= 0.55 "
          f"F1 {f3((ws['default'] or {}).get('f1'))}; best F1 {f3(top['f1'])} with wC={top['wC']}, wR={top['wR']}, "
          f"wA={top['wA']}, wV={top['wV']}, Medium >= {top['medium_threshold']}. {ws['note']}. Chart: `reports/confidence_bands.png`.\n")

    if trace:
        w("### 3.5 Traceability, access control and privacy\n")
        acc = trace["access_control"]
        w(f"* Analyses in the evaluation audit log: {trace['analyses_in_log']}. Recommendations where every displayed path "
          f"has >= 1 valid citation: {pct(trace['recommendations_every_path_cited'])}. Analyses with an evidence pack that "
          f"produced a valid citation: {pct(trace['analyses_with_pack_and_valid_citation'])}. Analyses without evidence (LLM "
          f"not called): {trace['analyses_without_pack_llm_not_called']}.")
        w(f"* Hallucinated IDs blocked: {trace['hallucinated_ids_blocked']}. Classifications with an explanation: "
          f"{pct(trace['classifications_with_explanation'])}. Analyses with full version info: {pct(trace['analyses_with_versions'])}.")
        w(f"* **PII patterns found in the audit database:** {trace['pii_patterns_in_audit_db']}.")
        w(f"* **Access control:** {len(acc['tickets'])} test tickets from the scenarios with the most Restricted history "
          f"({', '.join(acc['scenarios'])}) run as analyst1 (L1) and admin1 (L2_Admin): Restricted documents seen by L1 = "
          f"{acc['restricted_docs_seen_by_L1']}, by admin = {acc['restricted_docs_seen_by_admin']}.")
        pi = trace["prompt_injection"]
        w(f"* **Prompt injection** (\"ignore previous instructions ... disable MFA for all users\"): band {pi['band']}, "
          f"injected instruction followed: {pi['followed_injected_instruction']} (generator: {pi['llm']}).\n")

    if lat:
        w(f"### 3.6 Latency ({lat['hardware']})\n")
        for key in ("offline_generator", "api_llm"):
            if key in lat:
                n = lat[key].get("runs", lat["runs"])
                w(f"{key.replace('_', ' ')} (`{lat[key]['model']}`, {n} runs, generators {lat[key]['generators']}), ms:\n")
                w(md([{"stage": s, "p50": v["p50"], "p95": v["p95"]} for s, v in lat[key]["stages_ms"].items()]))

    w("## 4. Error analysis\n")
    w("Worst cases per dimension, tagged with a cause (full lists in the JSON files).\n")
    if cls:
        w(f"**Classification** ({cls['n_category_errors']} category and {cls['n_priority_errors']} priority errors on "
          f"{cls['n_standard']} standard tickets): {dict(Counter(tag_classification(e) for e in cls['errors']))}\n")
        w(md([{**e, "cause": tag_classification(e)} for e in cls["errors"]],
             ["ticket_id", "title", "kind", "label", "pred", "signals", "cause"]))
    if ret:
        bad = [x for x in ret["worst"] if x["mrr"] < 1]
        w(f"**Retrieval** ({ret['n_mrr_below_1']} queries whose first result is not relevant): "
          f"{dict(Counter(tag_retrieval(x) for x in bad))}\n")
        w(md([{**x, "cause": tag_retrieval(x)} for x in bad], ["ticket_id", "title", "mrr", "top1", "cause"]))
    if conf:
        w(f"**Confidence** (expected behaviour not met): {dict(Counter(tag_confidence(e) for e in conf['errors'])) or 'none'}\n")
        if conf["errors"]:
            w(md([{**e, "cause": tag_confidence(e)} for e in conf["errors"]], ["ticket_id", "group", "band", "score", "R", "title", "cause"]))
    if resl:
        wrong = [p for p in resl["per_ticket"] if p["path1_correct"] is False]
        w(f"**Resolution:** {len(wrong)} of {resl['n_tickets']} rating tickets had a path 1 that cites no qrels-relevant "
          "source" + (": " + ", ".join(p["ticket_id"] for p in wrong) if wrong else "") + ".\n")

    w("## 5. Known limitations\n")
    w("* **Synthetic, template-based data** (see the dataset's DATA_CARD): wording within a root cause is more regular than "
      "real tickets, so scores are optimistic; results are broken down by test type to show where the system struggles.")
    w("* **Priority** depends on explicit impact phrases, which the data card says only ~65% of tickets contain; tickets "
      "without a scope signal fall back to the model's text-based guess.")
    w("* **Small special sets** (25 novel, 15 vague, 10 contradictory; 25 for rating); English only.")
    w("* **Usefulness** is measured by automatic proxies (qrels-relevant citation, overlap with the reference fix) until two "
      "blind raters fill in `data/labels/ratings/`; the raters would be the builders, which the workflow flags as a bias.")
    w("* **Hand-set confidence weights** are kept for transparency; the sweep shows alternatives against a proxy target.")
    w("* The cross-encoder is a general MS MARCO model, not trained on IT tickets; Platt calibration fixes its scale, not its errors.")
    w("* PII masking uses the small spaCy model plus regex tuned to this dataset's formats; other name or host formats may need patterns.")
    w("* LLM: when no API key is configured or the API fails, analyses use the deterministic extractive generator; the "
      "generator actually used is reported per section above and stored in every audit row.")
    (REPORTS / "evaluation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "llm": config.llm_settings()["model"]}
    if cls:
        summary["classification"] = {"category_macro_f1": cls["category"]["macro_f1"],
                                      "subcategory_macro_f1": cls["subcategory"]["macro_f1"],
                                      "priority_macro_f1": cls["priority"]["macro_f1"],
                                      "p1_recall": cls["priority_recall"]["P1"], "p2_recall": cls["priority_recall"]["P2"]}
    if ret:
        summary["retrieval_hybrid_rerank"] = ret["by_mode"]["hybrid_rerank"]
    if resl:
        summary["resolution"] = {k: resl[k] for k in ("citation_pass_rate", "path1_relevant_rate",
                                                      "reference_overlap_mean", "hallucinated_ids_blocked")}
    if conf:
        summary["confidence"] = {"expected_behavior_success": conf["expected_behavior_success"],
                                 "false_escalation_rate_standard": conf["standard"]["false_escalation_rate_total"]}
    if trace:
        summary["access_control_restricted_seen_by_L1"] = trace["access_control"]["restricted_docs_seen_by_L1"]
    (REPORTS / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {REPORTS / 'evaluation_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run_scripts(a.llm)
    build_report()
