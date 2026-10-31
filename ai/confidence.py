"""Confidence scoring and fallback (Phase 7, FR-05).

score = 0.25 C + 0.35 R + 0.20 A + 0.20 V
  C  classification confidence  calibrated category probability
  R  retrieval strength         match score of the best evidence item
  A  evidence agreement         share of the top-3 evidence sources cited (validly) by path 1;
                                1.0 = all three agree, 0.33 = only one supports it
  V  citation validity          citation pass rate from validate.py

Hard rules are checked first and can force Low (or Escalate) regardless of the score.
Low never produces a confident answer: the analyst gets an escalation route and questions.
"""
from __future__ import annotations

from collections import Counter

from ai import config, dataset

BANDS = ("High", "Medium", "Low", "Escalate")


def _signals(classification: dict, pack: list[dict], validation: dict) -> dict:
    C = float(classification["category"]["confidence"])
    R = max((i["score"] for i in pack), default=0.0)
    top_sources = []
    for item in sorted(pack, key=lambda i: -i["score"]):
        if item["source_id"] not in top_sources:
            top_sources.append(item["source_id"])
    top_sources = top_sources[:config.AGREEMENT_TOP_N]
    paths = validation.get("paths", [])
    cited1 = {e["source_id"] for e in paths[0]["evidence"]} if paths else set()
    A = len(cited1 & set(top_sources)) / config.AGREEMENT_TOP_N
    V = float(validation.get("citation_pass_rate", 0.0))
    return {"C": round(C, 4), "R": round(float(R), 4), "A": round(A, 4), "V": round(V, 4)}


def _conflict_without_majority(validation: dict, pack: list[dict]) -> bool:
    conflicts = str(validation.get("conflicts") or "none").strip().lower()
    if conflicts in ("", "none", "no", "n/a", "no conflicts"):
        return False
    n_sources = len({i["source_id"] for i in pack}) or 1
    for p in validation.get("paths", []):
        if len({e["source_id"] for e in p["evidence"]}) / n_sources > 0.5:
            return False
    return True


def score(classification: dict, pack: list[dict], validation: dict, generation: dict, entities: dict) -> dict:
    s = _signals(classification, pack, validation)
    total = round(config.W_CLASSIFICATION * s["C"] + config.W_RETRIEVAL * s["R"]
                  + config.W_AGREEMENT * s["A"] + config.W_VALIDITY * s["V"], 4)
    reasons, rules = [], []

    escalate = classification["priority"]["label"] == "P1" or bool(entities.get("security"))
    if escalate:
        rules.append("escalate_p1_or_security")
        reasons.append("security incident: a human must own it" if entities.get("security")
                       else "P1 critical incident: a human must own it")

    low = False
    if not pack or s["R"] < config.RULE_MIN_RETRIEVAL:
        low = True
        rules.append("low_retrieval")
        reasons.append("no sufficiently similar resolved case")
    elif generation.get("status") == "invalid_output":
        low = True
        rules.append("invalid_llm_output")
        reasons.append("LLM output failed schema validation twice")
    if pack and s["V"] < config.RULE_MIN_VALIDITY:
        low = True
        rules.append("low_validity")
        reasons.append("recommendation not supported by sources")
    if pack and _conflict_without_majority(validation, pack):
        low = True
        rules.append("conflicting_evidence")
        reasons.append("historical fixes disagree")

    if escalate:
        band = "Escalate"
    elif low:
        band = "Low"
    elif total >= config.BAND_HIGH:
        band = "High"
    elif total >= config.BAND_MEDIUM:
        band = "Medium"
    else:
        band = "Low"
        reasons.append(f"overall score {total:.2f} below {config.BAND_MEDIUM}")

    if band == "High" and classification.get("uncertain"):
        band = "Medium"
        rules.append("uncertain_classification_cap")
        reasons.append("classification uncertain (close call): capped at Medium")
    if band == "Medium" and not any(r.startswith("classification uncertain") for r in reasons):
        weakest = min(s, key=s.get)
        reasons.append(f"verify before applying (weakest signal {weakest} = {s[weakest]:.2f})")
    return {"band": band, "score": total, "signals": s, "reasons": reasons, "rules_fired": rules,
            "weights": {"C": config.W_CLASSIFICATION, "R": config.W_RETRIEVAL,
                        "A": config.W_AGREEMENT, "V": config.W_VALIDITY}}


def fallback(confidence: dict, classification: dict, similar: list[dict], validation: dict, entities: dict) -> dict:
    """Escalation route, clarifying questions and partial evidence for Low / Escalate (section 10.4)."""
    cats, pol = dataset.categories(), dataset.policy()
    questions_by_sub = pol["clarifying_questions"]
    category = classification["category"]["label"]
    subcategory = classification["subcategory"]["label"]
    if confidence["band"] == "Escalate":
        if entities.get("security"):
            group, route = pol["security_group"], "on-call handoff (security incident)"
        else:
            group, route = cats[category]["default_group"], "on-call handoff (P1, category default group)"
    else:
        groups = [m["assignment_group"] for m in similar[:5] if m.get("assignment_group")]
        if groups:
            group, route = Counter(groups).most_common(1)[0][0], "most common group among top matches"
        else:
            group, route = cats[category]["default_group"], "category default group (taxonomy.yaml)"
    questions = [q for q in validation.get("missing_information") or [] if isinstance(q, str) and q.strip()]
    source = "llm_missing_information"
    if not questions:
        # a subcategory template is only trustworthy when the classification is not a close call
        trusted = not classification.get("uncertain") and subcategory in questions_by_sub
        questions = questions_by_sub[subcategory] if trusted else questions_by_sub["default"]
        source = f"template ({subcategory})" if trusted else "template (default: classification uncertain)"
    if classification.get("uncertain") and confidence["band"] != "Escalate" and not similar:
        route += " (classification uncertain: confirm category before routing)"
    related = [{"ticket_id": m["source_id"], "title": m["title"], "score": m["score"], "label": m["label"]}
               for m in similar[:5]]
    return {"band": confidence["band"], "escalation_group": group, "routing_basis": route,
            "clarifying_questions": questions[:5], "questions_source": source,
            "possibly_related": related, "reasons": confidence["reasons"]}
