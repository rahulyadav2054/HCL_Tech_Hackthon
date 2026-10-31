from ai import confidence
from ai.classify import apply_priority_rules


def cls(conf=0.9, priority="P3", uncertain=False, cat="Network", sub="VPN"):
    return {"category": {"label": cat, "confidence": conf, "runner_up": ["Access", 0.05]},
            "subcategory": {"label": sub, "confidence": 0.9},
            "priority": {"label": priority, "confidence": 0.8}, "uncertain": uncertain}


def pack(*scores):
    return [{"source_id": f"INC00000{i}", "score": s, "kind": "ticket"} for i, s in enumerate(scores)]


def val(cited=("INC000000",), rate=1.0, conflicts="none"):
    return {"paths": [{"rank": 1, "evidence": [{"source_id": c} for c in cited]}],
            "citation_pass_rate": rate, "conflicts": conflicts, "missing_information": []}


OK = {"status": "ok"}


def test_high_band_when_all_signals_strong():
    c = confidence.score(cls(), pack(0.95, 0.9, 0.85), val(("INC000000", "INC000001", "INC000002")), OK, {})
    assert c["band"] == "High" and c["signals"]["A"] == 1.0 and c["score"] >= 0.75


def test_agreement_one_of_three():
    c = confidence.score(cls(), pack(0.95, 0.9, 0.85), val(("INC000000",)), OK, {})
    assert abs(c["signals"]["A"] - 1 / 3) < 1e-3


def test_no_evidence_forces_low():
    c = confidence.score(cls(), [], {"paths": [], "citation_pass_rate": 0.0}, {"status": "skipped_no_evidence"}, {})
    assert c["band"] == "Low" and "no sufficiently similar resolved case" in c["reasons"]


def test_low_citation_validity_forces_low():
    c = confidence.score(cls(), pack(0.95), val(rate=0.4), OK, {})
    assert c["band"] == "Low" and "recommendation not supported by sources" in c["reasons"]


def test_conflict_without_majority_forces_low():
    v = {"paths": [{"rank": 1, "evidence": [{"source_id": "INC000000"}]},
                   {"rank": 2, "evidence": [{"source_id": "INC000001"}]}],
         "citation_pass_rate": 1.0, "conflicts": "two different root causes"}
    c = confidence.score(cls(), pack(0.9, 0.9, 0.9), v, OK, {})
    assert c["band"] == "Low" and "historical fixes disagree" in c["reasons"]


def test_p1_or_security_escalates_regardless_of_score():
    assert confidence.score(cls(priority="P1"), pack(0.99), val(), OK, {})["band"] == "Escalate"
    assert confidence.score(cls(), pack(0.99), val(), OK, {"security": True})["band"] == "Escalate"


def test_uncertain_classification_caps_at_medium():
    c = confidence.score(cls(uncertain=True), pack(0.95, 0.9, 0.85), val(("INC000000", "INC000001", "INC000002")), OK, {})
    assert c["band"] == "Medium"


def test_fallback_routes_and_asks_questions():
    c = confidence.score(cls(), [], {"paths": [], "citation_pass_rate": 0.0}, {"status": "skipped_no_evidence"}, {})
    similar = [{"source_id": "INC1", "title": "t", "score": 0.5, "label": "weak", "assignment_group": "Network-L3"}]
    fb = confidence.fallback(c, cls(), similar, {"missing_information": []}, {})
    assert fb["escalation_group"] == "Network-L3"
    assert fb["clarifying_questions"][0] == "Which VPN client version?"
    sec = confidence.fallback({"band": "Escalate", "reasons": []}, cls(), [], {}, {"security": True})
    assert sec["escalation_group"] == "Security-Ops"


def test_priority_rules():
    assert apply_priority_rules("P3", {"security": True}, "Production") == ("P1", "security keywords force P1")
    label, rule = apply_priority_rules("P3", {"scope": "all_users", "production": False}, "Production")
    assert label == "P2" and rule
    assert apply_priority_rules("P1", {"scope": "all_users"}, "Production") == ("P1", None)
    assert apply_priority_rules("P4", {"scope": "single_user"}, "Production") == ("P4", None)
