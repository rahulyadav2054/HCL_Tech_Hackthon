import pytest
from pydantic import ValidationError

from ai.preprocess import TicketIn, extract_entities, mask_pii, normalize, preprocess


def test_rejects_empty_title_or_description():
    with pytest.raises(ValidationError):
        TicketIn(title="  ", description="x")
    with pytest.raises(ValidationError):
        TicketIn(title="x", description="")


def test_missing_metadata_defaults_to_unknown():
    t = TicketIn(title="a", description="b", application=None, environment="")
    assert t.application == "unknown" and t.environment == "unknown"


@pytest.mark.parametrize("raw,secret", [
    ("Reach me at priya.sharma@corp.example.com today", "priya.sharma@corp.example.com"),
    ("Call me on +91 81234 56789.", "81234 56789"),
    ("Call +1 (415) 555-0134 please", "555-0134"),
    ("My laptop has IP 10.24.3.17 now", "10.24.3.17"),
    ("Machine LT-PUN-0423 is broken", "LT-PUN-0423"),
    ("Employee ID EMP123456.", "EMP123456"),
    ("username: vwang12", "vwang12"),
    ("Hi, this is Vikram Wang from Sales.", "Vikram Wang"),
    ("hi team, this is rahul rao from hr.", "rahul rao"),
    # formats used by the it-ticket-dataset
    ("david.taylor@acmecorp.example | +91 9107951635", "9107951635"),
    ("My laptop is LT-1831.", "LT-1831"),
    ("Backups on BKP-SRV-01 and DB-SRV-02 failed", "DB-SRV-02"),
    ("Printer PRN-HYD-3F-01 is offline", "PRN-HYD-3F-01"),
])
def test_pii_is_masked(raw, secret):
    masked, mapping = mask_pii(raw)
    assert secret not in masked
    assert any(secret in orig for _, orig in mapping)


def test_masking_keeps_error_codes_and_products():
    text = "Error 809 on GlobalProtect, ORA-12541, 554 5.2.2, DNS_PROBE_FINISHED_NXDOMAIN, set Chrome as default"
    masked, _ = mask_pii(text)
    for keep in ("809", "GlobalProtect", "ORA-12541", "554 5.2.2", "DNS_PROBE_FINISHED_NXDOMAIN", "Chrome"):
        assert keep in masked


def test_normalize_strips_html_reply_chain_and_signature():
    text = ("<p>VPN broken</p><br>still failing\n\nRegards,\nJohn Smith\n+1 555 123 4567")
    assert normalize(text) == "VPN broken still failing"
    chain = "Teams is down.\n\n-----Original Message-----\nFrom: A <a@b.com>\nSent: Monday\nold text"
    assert normalize(chain) == "Teams is down."


def test_entity_extraction():
    e = extract_entities("Since the firewall change I get error 809 and ORA-12541 and 0x80070005; "
                         "HTTP 503 for all users in production. Changed my password yesterday.")
    assert {"809", "ORA-12541", "0x80070005", "HTTP 503"} <= set(e["error_codes"])
    assert "after firewall change" in e["triggers"] and "after password change" in e["triggers"]
    assert e["scope"] == "all_users" and e["production"] is True


def test_scope_and_security_signals():
    assert extract_entities("Error 809 for the whole Pune office")["scope"] == "team"
    assert extract_entities("Only my laptop is affected")["scope"] == "single_user"
    assert extract_entities("I got MFA prompts I did not request, maybe compromised")["security"] is True


def test_preprocess_record_shape_and_title_weight():
    rec = preprocess({"title": "Cannot connect to VPN", "description": "error 809 call +91 81234 56789",
                      "application": "GlobalProtect VPN", "root_cause": "Stale cached credentials",
                      "resolution_notes": "Cleared Credential Manager"})
    assert rec["search_text"].count("cannot connect to vpn") >= 2
    assert "<PHONE>" in rec["description"]
    assert rec["resolution_text"].startswith("Root cause: Stale cached credentials. Fix:")
    assert rec["entities"]["error_codes"] == ["809"]
    assert rec["metadata"]["application"] == "GlobalProtect VPN"


def test_dataset_code_formats_match_kb():
    e = extract_entities("event 4740 in the logs, exit code 1603, Error 1040, ORA-01653, AADSTS50105")
    assert {"4740", "1603", "1040", "ORA-01653", "AADSTS50105"} <= set(e["error_codes"])


def test_dataset_signature_is_stripped():
    text = ("VPN broken. Just me, others are fine.\n\nRegards,\nDavid Taylor\n"
            "david.taylor@acmecorp.example | +91 9107951635")
    assert normalize(text) == "VPN broken. Just me, others are fine."
    assert extract_entities(text)["scope"] == "single_user"
